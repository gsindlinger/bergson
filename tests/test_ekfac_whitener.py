"""Unit tests for EkfacWhitener: validates H^power math against a synthetic
factor directory that bypasses the full gradient-collection pipeline."""

import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from bergson.hessians.ekfac_whitener import EkfacWhitener


def _make_synthetic_factor_dir(
    tmpdir: Path,
    module_shapes: dict[str, tuple[int, int]],
    seed: int = 0,
) -> Path:
    """Write a minimal EK-FAC factor directory with random orthonormal U_A, U_G
    and random positive eigenvalues Lambda."""
    torch.manual_seed(seed)
    factor_dir = tmpdir / "kfac"
    (factor_dir / "eigen_activation_sharded").mkdir(parents=True)
    (factor_dir / "eigen_gradient_sharded").mkdir(parents=True)
    (factor_dir / "eigenvalue_correction_sharded").mkdir(parents=True)

    eigen_a: dict[str, torch.Tensor] = {}
    eigen_g: dict[str, torch.Tensor] = {}
    lam: dict[str, torch.Tensor] = {}
    for name, (O, I) in module_shapes.items():
        eigen_a[name] = torch.linalg.qr(torch.randn(I, I))[0].float().contiguous()
        eigen_g[name] = torch.linalg.qr(torch.randn(O, O))[0].float().contiguous()
        lam[name] = (torch.rand(O, I).float() + 0.1).contiguous()  # strictly positive

    save_file(
        eigen_a, str(factor_dir / "eigen_activation_sharded" / "shard_0.safetensors")
    )
    save_file(
        eigen_g, str(factor_dir / "eigen_gradient_sharded" / "shard_0.safetensors")
    )
    save_file(
        lam, str(factor_dir / "eigenvalue_correction_sharded" / "shard_0.safetensors")
    )
    return factor_dir


def _apply_hessian_reference(
    grad_noi: torch.Tensor,
    U_A: torch.Tensor,
    U_G: torch.Tensor,
    lam: torch.Tensor,
    damp: float,
    power: float,
) -> torch.Tensor:
    """Reference implementation matching apply_hessian's math exactly."""
    g = grad_noi.clone().float()
    # Forward rotation: U_G^T @ g @ U_A
    g = U_G.T @ g
    g = g @ U_A
    # Damped eigenvalue scaling with mean(lam) damping (matches _hadamard).
    damped = lam + damp * lam.mean()
    g = g * damped.pow(power)
    # Rotate back
    g = U_G @ g
    g = g @ U_A.T
    return g


def test_whitener_power_minus_one_matches_reference():
    """Whitener with power=-1 must match the reference apply_hessian math."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shapes = {"mlp.0": (4, 6), "mlp.1": (8, 5)}
        factor_dir = _make_synthetic_factor_dir(tmp, shapes)

        whitener = EkfacWhitener(factor_dir, device="cpu", power=-1.0, damp=0.1)

        for name, (O, I) in shapes.items():
            torch.manual_seed(42)
            g = torch.randn(3, O, I)
            out = whitener.apply(name, g)
            # Build reference from scratch using loaded factors.
            ref_from_factors = _apply_hessian_reference(
                g,
                whitener.eigen_a[name],
                whitener.eigen_g[name],
                lam=_load_raw_lambda(factor_dir, name),
                damp=0.1,
                power=-1.0,
            )
            assert torch.allclose(
                out, ref_from_factors, atol=1e-5, rtol=1e-4
            ), f"[{name}] whitener power=-1 mismatch vs reference"


def _load_raw_lambda(factor_dir: Path, name: str) -> torch.Tensor:
    from safetensors.torch import load_file

    shard = factor_dir / "eigenvalue_correction_sharded" / "shard_0.safetensors"
    return load_file(str(shard))[name].float()


def test_whitener_sqrt_squared_equals_inverse():
    """(H^{-1/2})^2 g should equal H^{-1} g for the same factor dir + damp."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shapes = {"layer.0": (5, 7), "layer.1": (6, 6)}
        factor_dir = _make_synthetic_factor_dir(tmp, shapes, seed=1)

        half = EkfacWhitener(factor_dir, device="cpu", power=-0.5, damp=0.1)
        full = EkfacWhitener(factor_dir, device="cpu", power=-1.0, damp=0.1)

        for name, (O, I) in shapes.items():
            torch.manual_seed(name.__hash__() & 0xFFFF)
            g = torch.randn(2, O, I)
            out_twice = half.apply(name, half.apply(name, g))
            out_full = full.apply(name, g)
            assert torch.allclose(
                out_twice, out_full, atol=1e-4, rtol=1e-3
            ), f"[{name}] (H^-0.5)^2 != H^-1"


def test_whitener_passthrough_unknown_module():
    """Modules not in the factor dir must pass through unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        factor_dir = _make_synthetic_factor_dir(tmp, {"known": (3, 4)})
        whitener = EkfacWhitener(factor_dir, device="cpu", power=-0.5)
        g = torch.randn(2, 3, 4)
        assert torch.equal(whitener.apply("unknown", g), g)


def test_whitener_preserves_shape_dtype_device():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        factor_dir = _make_synthetic_factor_dir(tmp, {"m": (4, 4)})
        whitener = EkfacWhitener(factor_dir, device="cpu", power=-0.5)
        g = torch.randn(5, 4, 4, dtype=torch.float64)
        out = whitener.apply("m", g)
        assert out.shape == g.shape
        assert out.dtype == g.dtype
        assert out.device == g.device


def test_whitener_token_level_shape():
    """[N, S, O, I] input (attribute_tokens path) should also work."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        factor_dir = _make_synthetic_factor_dir(tmp, {"m": (4, 5)})
        whitener = EkfacWhitener(factor_dir, device="cpu", power=-0.5)
        g = torch.randn(2, 7, 4, 5)
        out = whitener.apply("m", g)
        assert out.shape == g.shape


def _make_uncorrected_factor_dir(
    tmpdir: Path,
    module_shapes: dict[str, tuple[int, int]],
    total_processed: int = 1000,
    seed: int = 0,
) -> Path:
    """Write a factor directory with no eigenvalue_correction_sharded/, only
    the per-side eigenvalues that plain K-FAC (ev_correction=False) writes."""
    torch.manual_seed(seed)
    factor_dir = tmpdir / "kfac"
    (factor_dir / "eigen_activation_sharded").mkdir(parents=True)
    (factor_dir / "eigen_gradient_sharded").mkdir(parents=True)
    (factor_dir / "eigval_activation_sharded").mkdir(parents=True)
    (factor_dir / "eigval_gradient_sharded").mkdir(parents=True)

    eigen_a: dict[str, torch.Tensor] = {}
    eigen_g: dict[str, torch.Tensor] = {}
    eigval_a: dict[str, torch.Tensor] = {}
    eigval_g: dict[str, torch.Tensor] = {}
    for name, (O, I) in module_shapes.items():
        eigen_a[name] = torch.linalg.qr(torch.randn(I, I))[0].float().contiguous()
        eigen_g[name] = torch.linalg.qr(torch.randn(O, O))[0].float().contiguous()
        eigval_a[name] = (torch.rand(I).float() + 0.1).contiguous()
        eigval_g[name] = (torch.rand(O).float() + 0.1).contiguous()

    save_file(
        eigen_a, str(factor_dir / "eigen_activation_sharded" / "shard_0.safetensors")
    )
    save_file(
        eigen_g, str(factor_dir / "eigen_gradient_sharded" / "shard_0.safetensors")
    )
    save_file(
        eigval_a, str(factor_dir / "eigval_activation_sharded" / "shard_0.safetensors")
    )
    save_file(
        eigval_g, str(factor_dir / "eigval_gradient_sharded" / "shard_0.safetensors")
    )
    torch.save(torch.tensor(total_processed), factor_dir / "total_processed.pt")
    return factor_dir


def test_whitener_uncorrected_fallback_matches_outer_product():
    """With no eigenvalue_correction_sharded/ (plain K-FAC), the whitener
    must fall back to Lambda[o,i] = E_G[o] * E_A[i] * total_processed."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shapes = {"mlp.0": (4, 6), "mlp.1": (8, 5)}
        total_processed = 12345
        factor_dir = _make_uncorrected_factor_dir(tmp, shapes, total_processed)

        whitener = EkfacWhitener(factor_dir, device="cpu", power=-1.0, damp=0.1)

        from safetensors.torch import load_file

        eigval_a = load_file(
            str(factor_dir / "eigval_activation_sharded" / "shard_0.safetensors")
        )
        eigval_g = load_file(
            str(factor_dir / "eigval_gradient_sharded" / "shard_0.safetensors")
        )

        for name, (O, I) in shapes.items():
            expected_lambda = (
                torch.outer(eigval_g[name], eigval_a[name]) * total_processed
            )
            torch.manual_seed(7)
            g = torch.randn(3, O, I)
            out = whitener.apply(name, g)
            ref = _apply_hessian_reference(
                g,
                whitener.eigen_a[name],
                whitener.eigen_g[name],
                lam=expected_lambda,
                damp=0.1,
                power=-1.0,
            )
            assert torch.allclose(
                out, ref, atol=1e-4, rtol=1e-3
            ), f"[{name}] uncorrected fallback mismatch vs outer-product reference"


def test_whitener_prefers_real_correction_when_present():
    """If eigenvalue_correction_sharded/ exists, it must be used even when
    eigval_*_sharded are also present (e.g. left over from a kfac run that
    later got an ekfac correction pass added)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shapes = {"m": (4, 4)}
        factor_dir = _make_synthetic_factor_dir(tmp, shapes)
        # Also drop in eigval_*_sharded with very different values, to prove
        # they're ignored when the real correction is available.
        (factor_dir / "eigval_activation_sharded").mkdir(parents=True)
        (factor_dir / "eigval_gradient_sharded").mkdir(parents=True)
        save_file(
            {"m": torch.full((4,), 999.0)},
            str(factor_dir / "eigval_activation_sharded" / "shard_0.safetensors"),
        )
        save_file(
            {"m": torch.full((4,), 999.0)},
            str(factor_dir / "eigval_gradient_sharded" / "shard_0.safetensors"),
        )

        whitener = EkfacWhitener(factor_dir, device="cpu", power=-1.0, damp=0.1)
        g = torch.randn(2, 4, 4)
        out = whitener.apply("m", g)
        ref = _apply_hessian_reference(
            g,
            whitener.eigen_a["m"],
            whitener.eigen_g["m"],
            lam=_load_raw_lambda(factor_dir, "m"),
            damp=0.1,
            power=-1.0,
        )
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
