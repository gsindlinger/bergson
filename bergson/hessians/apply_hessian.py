import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from simple_parsing import ArgumentParser
from torch import Tensor

from bergson.collector.collector import create_projection_matrix
from bergson.data import create_index, load_gradients
from bergson.hessians.eigenvectors import (
    _compute_full_matrix,
    fair_distribute_by_cost,
)
from bergson.hessians.sharded_computation import ShardedMul
from bergson.utils.logger import get_logger
from bergson.utils.utils import get_device


@dataclass
class EkfacConfig:
    hessian_method_path: str
    gradient_path: str
    run_path: str
    debug: bool = False
    lambda_damp_factor: float = 0.1
    query_chunk_size: int = 0
    projection_dim: int = 0
    projection_type: Literal["normal", "rademacher"] = "rademacher"


SIDE_TO_COV = {"left": "gradient", "right": "activation"}


def build_kfac_projections(
    hessian_method_path: str,
    projection_dim: int,
    projection_type: Literal["normal", "rademacher"],
    lambda_damp_factor: float,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Build and save ``M = R · cov^{-1/2}`` (precondition+sketch) per side.

    For each module and side (left=gradient, right=activation):
        M = R · Q · diag((E + λ·mean(E))^{-1/2}) · Qᵀ   [p, d]

    Saved to ``hessian_method_path/projection_{side}_sharded/``.
    Loaded at collection time via ``load_kfac_projections`` to short-circuit
    the random-projection call and apply M directly.
    """
    if projection_dim <= 0:
        raise ValueError(
            f"build_kfac_projections requires projection_dim > 0; got {projection_dim}."
        )

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    side_dims: dict[str, dict[str, int]] = {"left": {}, "right": {}}
    for side, cov in SIDE_TO_COV.items():
        with safe_open(
            os.path.join(
                hessian_method_path, f"eigval_{cov}_sharded/shard_0.safetensors"
            ),
            framework="pt",
        ) as f:
            for name in f.keys():
                side_dims[side][name] = f.get_tensor(name).shape[-1]

    names = list(side_dims["left"].keys())
    per_layer_dim = {n: max(side_dims["left"][n], side_dims["right"][n]) for n in names}
    my_names = fair_distribute_by_cost(per_layer_dim, world_size)[rank]

    out_dirs = {
        side: os.path.join(hessian_method_path, f"projection_{side}_sharded")
        for side in ("left", "right")
    }
    if rank == 0:
        for d in out_dirs.values():
            os.makedirs(d, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    saved: dict[str, dict[str, Tensor]] = {"left": {}, "right": {}}
    for name in my_names:
        for side, cov in SIDE_TO_COV.items():
            d = side_dims[side][name]
            Q = _compute_full_matrix(
                name=name,
                shard_path=os.path.join(hessian_method_path, f"eigen_{cov}_sharded"),
                rank=rank,
                world_size=world_size,
            ).to(device=device, dtype=torch.float32)
            E = _compute_full_matrix(
                name=name,
                shard_path=os.path.join(hessian_method_path, f"eigval_{cov}_sharded"),
                rank=rank,
                world_size=world_size,
            ).to(device=device, dtype=torch.float32)

            damp = lambda_damp_factor * E.mean()
            D = (E + damp).clamp_min(torch.finfo(torch.float32).tiny).rsqrt()

            R = create_projection_matrix(
                f"{name}/{side}",
                projection_dim,
                d,
                torch.float32,
                device,
                projection_type,
            )
            # Unbias: create_projection_matrix returns unit-norm rows so
            # E[RᵀR] = (p/d)·I; rescale to E[RᵀR] = I to avoid per-layer
            # p²/(d_left·d_right) score reweighting.
            R = R * (d / projection_dim) ** 0.5

            M = R @ (Q * D) @ Q.T
            saved[side][name] = M.to(dtype=dtype).cpu().contiguous()

    for side in ("left", "right"):
        save_file(
            saved[side],
            os.path.join(out_dirs[side], f"shard_{rank}.safetensors"),
        )

    get_logger().info(
        f"Saved M_left/M_right to {out_dirs['left']} and {out_dirs['right']}"
    )

    if dist.is_initialized():
        dist.barrier()


def load_kfac_projections(
    hessian_method_path: str | os.PathLike,
    cache: dict,
    target_names,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Populate ``cache`` with M matrices saved by ``build_kfac_projections``.

    ``cache`` is ``processor._projection_matrices`` (keyed by ``(name, side, device)``);
    a cache hit short-circuits random projection during gradient collection,
    so gradients come out preconditioned-and-sketched in a single matmul.
    """
    targets = set(target_names)
    for side in ("left", "right"):
        side_dir = Path(hessian_method_path) / f"projection_{side}_sharded"
        if not side_dir.exists():
            return
        for shard_file in sorted(side_dir.glob("shard_*.safetensors")):
            with safe_open(str(shard_file), framework="pt", device=str(device)) as f:
                for name in f.keys():
                    if name in targets:
                        cache[(name, side, device)] = f.get_tensor(name).to(dtype=dtype)


class EkfacApplicator:
    def __init__(self, cfg: EkfacConfig):
        self.cfg = cfg
        self.path = cfg.hessian_method_path
        self.gradient_path = cfg.gradient_path

        self.logger = get_logger(
            "EkfacApplicator", level="DEBUG" if cfg.debug else "INFO"
        )

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = f"cuda:{self.rank}"

        self.sharded_computer = ShardedMul()

    def compute_ivhp_sharded(self):
        if self.cfg.projection_dim > 0:
            return self._apply_compressed()
        return self._apply_legacy()

    def _apply_compressed(self):
        """Apply M_left · G_q · M_rightᵀ to the saved query gradients.

        Assumes ``build_kfac_projections`` has already saved M under
        ``hessian_method_path/projection_{side}_sharded/``.
        Output shape per layer is [N, p, p].
        """
        p = self.cfg.projection_dim

        M_left: dict[str, Tensor] = {}
        M_right: dict[str, Tensor] = {}
        for side, store in (("left", M_left), ("right", M_right)):
            side_dir = os.path.join(self.path, f"projection_{side}_sharded")
            for shard_file in sorted(Path(side_dir).glob("shard_*.safetensors")):
                shard = load_file(str(shard_file), device=str(self.device))
                for k, v in shard.items():
                    store[k] = v.to(dtype=torch.float32)

        mmap = load_gradients(self.gradient_path)
        with open(os.path.join(self.gradient_path, "info.json")) as f:
            info = json.load(f)

        grad_sizes = {name: p * p for name in M_left}
        grad_buffer = create_index(
            Path(self.cfg.run_path),
            num_grads=info["num_grads"],
            grad_sizes=grad_sizes,
            dtype=np.float32,
        )

        self.logger.info(
            f"Loaded gradients for {len(mmap)} queries, applying M·G·Mᵀ..."
        )

        for name, M_l in M_left.items():
            M_r = M_right[name]
            d_S, d_A = M_l.shape[1], M_r.shape[1]
            G = (
                torch.from_numpy(mmap[name][:])
                .to(device=self.device, dtype=torch.float32)
                .view(-1, d_S, d_A)
            )
            # ĝ_q = M_left · G · M_rightᵀ  [N, p, p]
            sketched = torch.einsum("ps,nsa,ra->npr", M_l, G, M_r)
            grad_buffer[name][:] = (
                sketched.to(device="cpu", non_blocking=True).flatten(1).numpy()
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        grad_buffer.flush()
        self.logger.info(f"Saved sketched IVHP gradients to {self.cfg.run_path}")

    def _apply_legacy(self):
        """Full-rank IVHP via the eigenbasis rotate-divide-rotate path."""
        eigen_a = load_file(
            self.path + f"/eigen_activation_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )
        eigen_g = load_file(
            self.path + f"/eigen_gradient_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )
        lambda_factor = load_file(
            self.path + f"/eigenvalue_correction_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )

        for k, v in lambda_factor.items():
            eigen_a[k] = eigen_a[k].to(dtype=torch.float32)
            eigen_g[k] = eigen_g[k].to(dtype=torch.float32)
            lambda_factor[k] = v.to(dtype=torch.float32)

        grad_sizes = {
            name: eigen_g[name].shape[1] * eigen_a[name].shape[1] for name in eigen_a
        }
        grad_shapes = {
            name: [eigen_g[name].shape[1], eigen_a[name].shape[1]] for name in eigen_a
        }

        mmap = load_gradients(self.gradient_path)
        with open(os.path.join(self.gradient_path, "info.json")) as f:
            info = json.load(f)

        num_grads = info["num_grads"]
        grad_buffer = create_index(
            Path(self.cfg.run_path),
            num_grads=num_grads,
            grad_sizes=grad_sizes,
            dtype=np.float32,
            grad_shapes=grad_shapes,
        )

        self.logger.info(
            f"Loaded gradients for {len(mmap)} queries and computing IVHP..."
        )

        query_chunk_size = self.cfg.query_chunk_size or num_grads
        query_chunk_size = max(1, min(query_chunk_size, num_grads))

        with torch.inference_mode():
            for k in eigen_a:
                self.logger.debug("Computing IVHP for %s", k)
                out_dim = eigen_g[k].shape[1]
                in_dim = eigen_a[k].shape[1]

                for start in range(0, num_grads, query_chunk_size):
                    end = min(start + query_chunk_size, num_grads)

                    gradients_noi = torch.from_numpy(mmap[k][start:end].copy()).to(
                        device=self.device, dtype=torch.float32
                    )
                    gradients_noi = gradients_noi.view(-1, out_dim, in_dim)

                    # Forward rotation into eigenbasis: Q_S^T @ G @ Q_A
                    transformed = self.sharded_computer._matmul(
                        vector_nsa=gradients_noi, matrix_cb=eigen_a[k]
                    )
                    del gradients_noi

                    transformed = self.sharded_computer._matmul(
                        vector_nsa=transformed.transpose(-2, -1),
                        matrix_cb=eigen_g[k],
                    ).transpose(-2, -1)

                    # Divide by damped eigenvalues in eigenbasis.
                    self.sharded_computer._hadamard(
                        matrix_noi=transformed,
                        lambda_ci=lambda_factor[k],
                        lambda_damp_factor=self.cfg.lambda_damp_factor,
                    )

                    # Rotate back to parameter space: Q_S @ G' @ Q_A^T
                    transformed = self.sharded_computer._transpose_matmul(
                        vector_nsa=transformed.transpose(-2, -1),
                        matrix_cb=eigen_g[k],
                    ).transpose(-2, -1)

                    transformed = self.sharded_computer._transpose_matmul(
                        vector_nsa=transformed,
                        matrix_cb=eigen_a[k],
                    )

                    grad_buffer[k][start:end] = (
                        transformed.to(device="cpu", non_blocking=True)
                        .flatten(1)
                        .numpy()
                    )
                    del transformed

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.logger.debug("Finished H^{-1} G = Q_S @ (G' / lambda) @ Q_A^T")
        del eigen_a, eigen_g, lambda_factor
        gc.collect()

        torch.cuda.synchronize()

        grad_buffer.flush()

        self.logger.info(f"Saved IVHP gradients to {self.cfg.run_path}")


def apply_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    cfg: EkfacConfig,
):
    """Worker function for distributed IVHP computation."""
    from datetime import timedelta

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{local_rank}"),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )

    applicator = EkfacApplicator(cfg)
    applicator.compute_ivhp_sharded()


def build_projections_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    cfg: EkfacConfig,
):
    """Worker for building ``M = R · cov^{-1/2}`` and saving to disk."""
    from datetime import timedelta

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{local_rank}"),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )

    build_kfac_projections(
        cfg.hessian_method_path,
        projection_dim=cfg.projection_dim,
        projection_type=cfg.projection_type,
        lambda_damp_factor=cfg.lambda_damp_factor,
        dtype=torch.float32,
        device=torch.device(get_device(rank)),
    )


if __name__ == "__main__":
    from bergson.config import DistributedConfig
    from bergson.distributed import launch_distributed_run

    parser = ArgumentParser()
    parser.add_arguments(EkfacConfig, dest="cfg")
    args = parser.parse_args()

    launch_distributed_run(
        "apply_hessian",
        apply_worker,
        [args.cfg],
        DistributedConfig(),
    )
