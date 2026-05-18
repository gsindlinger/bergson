"""EK-FAC whitener: applies H^power in-place to per-sample gradients in their
native [N, O, I] shape, using the layer-local Kronecker eigendecomposition
produced by the EK-FAC Hessian fit.

The same math powers ``bergson.hessians.apply_hessian`` with power=-1; here we
parametrize the exponent so the same kernel computes H^{-1/2} (the whitening operator) and discards the disk round-trip by being applied
on-the-fly inside the gradient-collection hook.

Factor directory layout (produced by ``approximate_hessians``):
    <method_dir>/eigen_activation_sharded/shard_<r>.safetensors     -> U_A[name]
    <method_dir>/eigen_gradient_sharded/shard_<r>.safetensors       -> U_G[name]
    <method_dir>/eigenvalue_correction_sharded/shard_<r>.safetensors -> Lambda[name]

Shards partition modules across ranks, so we concatenate dicts across all
shards at load time to get a single per-module mapping.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file


class EkfacWhitener:
    """Applies H^power to per-sample gradients using EK-FAC Kronecker factors.

    Parameters
    ----------
    factor_dir : str | Path
        Directory containing ``eigen_activation_sharded/``,
        ``eigen_gradient_sharded/``, and ``eigenvalue_correction_sharded/``
        subdirectories with per-rank safetensors shards.
    device : torch.device | str
        Device to load factors onto.
    power : float
        Matrix power to apply. Standard choices:
          -0.5 : H^{-1/2}, two-sided whitening.
          -1.0 : H^{-1}, one-sided (equivalent to existing apply_hessian).
    damp : float
        Damping factor added to eigenvalues before exponentiation:
        scale = (lambda + damp * mean(lambda))^power, where ``mean(lambda)``
        is taken over non-zero entries per module (matches the convention in
        ``sharded_computation._hadamard``).
    dtype : torch.dtype
        Compute dtype for the factors. float32 recommended.
    """

    def __init__(
        self,
        factor_dir: str | Path,
        device: torch.device | str,
        power: float = -0.5,
        damp: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.power = power
        self.damp = damp
        self.dtype = dtype

        eigen_a, eigen_g, lambda_factor = self._load_all_shards(Path(factor_dir))

        self.eigen_a: dict[str, torch.Tensor] = {
            k: v.to(device=self.device, dtype=self.dtype) for k, v in eigen_a.items()
        }
        self.eigen_g: dict[str, torch.Tensor] = {
            k: v.to(device=self.device, dtype=self.dtype) for k, v in eigen_g.items()
        }
        self.scale: dict[str, torch.Tensor] = {}
        for name, lam in lambda_factor.items():
            lam = lam.to(device=self.device, dtype=self.dtype)
            # Matches _hadamard convention in sharded_computation.py so H^{-1}
            # numerics agree exactly with apply_hessian.
            damped = lam + self.damp * lam.mean()
            self.scale[name] = damped.pow(self.power)

    @staticmethod
    def _load_all_shards(
        factor_dir: Path,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        eigen_a: dict[str, torch.Tensor] = {}
        eigen_g: dict[str, torch.Tensor] = {}
        lambda_factor: dict[str, torch.Tensor] = {}
        for shard in sorted(
            (factor_dir / "eigen_activation_sharded").glob("shard_*.safetensors")
        ):
            eigen_a.update(load_file(str(shard)))
        for shard in sorted(
            (factor_dir / "eigen_gradient_sharded").glob("shard_*.safetensors")
        ):
            eigen_g.update(load_file(str(shard)))
        for shard in sorted(
            (factor_dir / "eigenvalue_correction_sharded").glob("shard_*.safetensors")
        ):
            lambda_factor.update(load_file(str(shard)))
        if not eigen_a:
            raise FileNotFoundError(
                f"No EK-FAC factor shards found under {factor_dir}. Expected "
                f"eigen_activation_sharded/shard_*.safetensors etc."
            )
        return eigen_a, eigen_g, lambda_factor

    def has_module(self, name: str) -> bool:
        return name in self.eigen_a

    def modules(self) -> set[str]:
        return set(self.eigen_a.keys())

    def apply(self, name: str, grad_noi: torch.Tensor) -> torch.Tensor:
        """Apply H[name]^power to a batch of per-sample gradients in-shape.

        Parameters
        ----------
        name : str
            Module name (key in the factor directory).
        grad_noi : torch.Tensor
            Per-sample gradients, shape ``[..., O, I]``. Typically ``[N, O, I]``
            from a standard backward hook, or ``[N, S, O, I]`` for token-level.

        Returns
        -------
        torch.Tensor
            ``H[name]^power @ grad`` in the same shape and dtype as ``grad_noi``.
        """
        if name not in self.eigen_a:
            # Module not in factor dir (e.g. lm_head filtered). Pass through.
            return grad_noi

        orig_dtype = grad_noi.dtype
        orig_device = grad_noi.device
        U_A = self.eigen_a[name]
        U_G = self.eigen_g[name]
        scale = self.scale[name]  # [O, I] in eigenbasis

        # Cast once for numerical stability.
        g = grad_noi.to(device=self.device, dtype=self.dtype)

        # Forward rotation: U_G^T @ g @ U_A  (still [..., O, I])
        g = torch.matmul(U_G.transpose(-1, -2), g)
        g = torch.matmul(g, U_A)

        # Diagonal scaling in the eigenbasis: (lambda + damp)^power elementwise
        g = g * scale

        # Rotate back: U_G @ (...) @ U_A^T
        g = torch.matmul(U_G, g)
        g = torch.matmul(g, U_A.transpose(-1, -2))

        return g.to(device=orig_device, dtype=orig_dtype)
