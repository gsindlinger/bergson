"""Integration test: GradientCollector + EK-FAC whitener produces the expected
preconditioned-and-sketched per-sample gradient.

Reference (math):
    Given per-sample g_out  [N, S, O]  and per-sample a_in  [N, S, I],
    let P  = g_out.mT @ a_in                              [N, O, I]
        P' = whitener.apply(name, P)                      (rotate → scale → rotate)
        P'' = G_proj @ P' @ A_proj.T                      [N, p, p]

    The collector with projection_dim=p, include_bias=False, and
    ekfac_whitener_path=<factor dir> must produce P'' in its mod_grads.
"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from datasets import Dataset
from safetensors.torch import save_file

from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import IndexConfig
from bergson.gradients import GradientProcessor
from bergson.hessians.ekfac_whitener import EkfacWhitener


class _TwoLayerMLP(nn.Module):
    def __init__(self, I: int, H: int, O: int):
        super().__init__()
        self.fc1 = nn.Linear(I, H, bias=False)
        self.fc2 = nn.Linear(H, O, bias=False)

    @property
    def device(self):
        return self.fc1.weight.device

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _write_synthetic_factors(
    factor_dir: Path, shapes: dict[str, tuple[int, int]], seed: int = 0
) -> None:
    """Emit shard_0.safetensors files with random orthogonal U_A, U_G and
    positive eigenvalues Lambda for each module."""
    torch.manual_seed(seed)
    (factor_dir / "eigen_activation_sharded").mkdir(parents=True, exist_ok=True)
    (factor_dir / "eigen_gradient_sharded").mkdir(parents=True, exist_ok=True)
    (factor_dir / "eigenvalue_correction_sharded").mkdir(parents=True, exist_ok=True)

    eigen_a, eigen_g, lam = {}, {}, {}
    for name, (O, I) in shapes.items():
        eigen_a[name] = torch.linalg.qr(torch.randn(I, I))[0].float().contiguous()
        eigen_g[name] = torch.linalg.qr(torch.randn(O, O))[0].float().contiguous()
        lam[name] = (torch.rand(O, I).float() + 0.1).contiguous()

    save_file(eigen_a, str(factor_dir / "eigen_activation_sharded" / "shard_0.safetensors"))
    save_file(eigen_g, str(factor_dir / "eigen_gradient_sharded" / "shard_0.safetensors"))
    save_file(lam, str(factor_dir / "eigenvalue_correction_sharded" / "shard_0.safetensors"))


def test_collector_whitener_matches_manual_sketch():
    """Collector output with whitener + projection must equal the hand-computed
    sketch G_proj @ (H^{-1/2} · P) @ A_proj.T on a two-layer MLP."""
    torch.manual_seed(0)
    N, S, I, H, O = 2, 4, 5, 7, 3
    p = 4
    damp = 0.1

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        factor_dir = tmp / "kfac"
        _write_synthetic_factors(
            factor_dir,
            {"fc1": (H, I), "fc2": (O, H)},
            seed=7,
        )

        model = _TwoLayerMLP(I, H, O)
        data = Dataset.from_dict({"input_ids": [[1] * 10] * N})

        cfg = IndexConfig(
            run_path=str(tmp / "run"),
            skip_index=True,
            skip_preconditioners=True,
            projection_dim=p,
            ekfac_whitener_path=str(factor_dir),
            ekfac_whitener_damp=damp,
        )
        processor = GradientProcessor(projection_dim=p, include_bias=False)
        collector = GradientCollector(
            model=model,
            cfg=cfg,
            data=data,
            processor=processor,
            target_modules={"fc1", "fc2"},
        )

        x = torch.randn(N, S, I)
        with collector:
            model.zero_grad()
            out = model(x)
            loss = (out ** 2).sum()
            loss.backward()
            collected = {k: v.clone() for k, v in collector.mod_grads.items()}

        # Independent reference whitener (loaded fresh from the same factor dir)
        ref_whitener = EkfacWhitener(
            factor_dir, device="cpu", power=-0.5, damp=damp
        )

        # Re-run the forward/backward *without* the whitener to capture raw
        # per-sample activations and grad_outputs per layer.
        raw_a = {}
        raw_g = {}

        def make_fwd_hook(name):
            def hook(module, inp, _):
                raw_a[name] = inp[0].detach().clone()
            return hook

        def make_bwd_hook(name):
            def hook(module, _, grad_out):
                raw_g[name] = grad_out[0].detach().clone()
            return hook

        handles = []
        for name in ("fc1", "fc2"):
            layer = model.get_submodule(name)
            handles.append(layer.register_forward_hook(make_fwd_hook(name)))
            handles.append(layer.register_full_backward_hook(make_bwd_hook(name)))
        try:
            model.zero_grad()
            (model(x) ** 2).sum().backward()
        finally:
            for h in handles:
                h.remove()

        for name, (out_dim, in_dim) in [("fc1", (H, I)), ("fc2", (O, H))]:
            a = raw_a[name]  # [N, S, in_dim]
            g_out = raw_g[name]  # [N, S, out_dim]
            # Per-sample outer product [N, out_dim, in_dim]
            P_full = g_out.mT @ a

            # Reference: apply whitener, then double-sided project with the
            # *same* projection matrices the collector used.
            P_white = ref_whitener.apply(name, P_full)
            G_proj = collector.projection(name, p, out_dim, "left", P_white.device, P_white.dtype)
            A_proj = collector.projection(name, p, in_dim, "right", P_white.device, P_white.dtype)
            # double_sided_projection = G_proj [p,O] @ P [N,O,I] @ A_proj.T [I,p]
            P_ref = G_proj @ P_white @ A_proj.T

            # `collected[name]` is flattened to [N, p*p]; reshape and compare.
            collected_mat = collected[name].view(N, p, p)

            torch.testing.assert_close(
                collected_mat, P_ref, atol=1e-4, rtol=1e-3,
                msg=f"[{name}] collector output != manual sketch",
            )
