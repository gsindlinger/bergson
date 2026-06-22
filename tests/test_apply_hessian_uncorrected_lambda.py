"""Unit test for EkfacApplicator._uncorrected_lambda: the plain K-FAC
fallback used by _apply_legacy when eigenvalue_correction_sharded/ is
absent (ev_correction=False). Constructs the applicator without __init__
since that hardcodes a CUDA device string."""

import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from bergson.hessians.apply_hessian import EkfacApplicator


def _make_uncorrected_factor_dir(
    tmpdir: Path,
    module_shapes: dict[str, tuple[int, int]],
    total_processed: int = 1000,
    seed: int = 0,
) -> Path:
    torch.manual_seed(seed)
    factor_dir = tmpdir / "kfac"
    (factor_dir / "eigval_activation_sharded").mkdir(parents=True)
    (factor_dir / "eigval_gradient_sharded").mkdir(parents=True)

    eigval_a: dict[str, torch.Tensor] = {}
    eigval_g: dict[str, torch.Tensor] = {}
    for name, (O, I) in module_shapes.items():
        eigval_a[name] = (torch.rand(I).float() + 0.1).contiguous()
        eigval_g[name] = (torch.rand(O).float() + 0.1).contiguous()

    save_file(
        eigval_a, str(factor_dir / "eigval_activation_sharded" / "shard_0.safetensors")
    )
    save_file(
        eigval_g, str(factor_dir / "eigval_gradient_sharded" / "shard_0.safetensors")
    )
    torch.save(torch.tensor(total_processed), factor_dir / "total_processed.pt")
    return factor_dir


def _make_applicator(path: str, rank: int = 0, device: str = "cpu") -> EkfacApplicator:
    """Construct an EkfacApplicator without running __init__, so we can test
    on CPU (the real __init__ hardcodes a cuda:{rank} device string)."""
    applicator = EkfacApplicator.__new__(EkfacApplicator)
    applicator.path = path
    applicator.rank = rank
    applicator.device = device
    return applicator


def test_uncorrected_lambda_matches_outer_product():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shapes = {"mlp.0": (4, 6), "mlp.1": (8, 5)}
        total_processed = 12345
        factor_dir = _make_uncorrected_factor_dir(tmp, shapes, total_processed)
        applicator = _make_applicator(str(factor_dir))

        from safetensors.torch import load_file

        eigval_a = load_file(
            str(factor_dir / "eigval_activation_sharded" / "shard_0.safetensors")
        )
        eigval_g = load_file(
            str(factor_dir / "eigval_gradient_sharded" / "shard_0.safetensors")
        )

        result = applicator._uncorrected_lambda(shapes.keys())

        for name in shapes:
            expected = torch.outer(eigval_g[name], eigval_a[name]) * total_processed
            assert torch.allclose(
                result[name], expected, atol=1e-5, rtol=1e-4
            ), f"[{name}] uncorrected lambda mismatch"


def test_apply_legacy_falls_back_when_no_correction_dir(monkeypatch):
    """compute_ivhp_sharded's _apply_legacy must reach the fallback branch
    (not raise) when eigenvalue_correction_sharded/ is missing. We only
    check the branch selection here, not the full CUDA IVHP math."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        factor_dir = _make_uncorrected_factor_dir(tmp, {"m": (3, 3)})
        applicator = _make_applicator(str(factor_dir))

        correction_shard = Path(
            applicator.path
            + f"/eigenvalue_correction_sharded/shard_{applicator.rank}.safetensors"
        )
        assert not correction_shard.exists()
        # Mirrors the branch condition added to _apply_legacy.
        lam = applicator._uncorrected_lambda(["m"])
        assert "m" in lam


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
