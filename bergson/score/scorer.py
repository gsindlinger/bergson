from collections.abc import Callable

import torch
from torch import Tensor

from bergson.score.score_writer import ScoreWriter


def _compute_low_rank_factors(
    query_grads: dict[str, Tensor],
    modules: list[str],
    grad_shapes: dict[str, list[int]],
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Approximate each module's query gradients with a truncated SVD.

    For module *m* with weight shape ``(O, I)``, each query gradient is a flat
    vector of length ``O * I``.  Reshaped to ``[n_queries, O, I]`` the stack has
    rank bounded by the number of tokens in the query sequences.  A rank-*r*
    truncated SVD gives ``U [n_queries, O, r]`` and ``V [n_queries, I, r]``
    (with singular values absorbed into ``U``) such that the original gradient
    matrix is approximated by ``U @ V^T``.

    Returns ``{module: (U, V)}`` stored on *device* in *dtype*.

    Memory during scoring is ``n_queries * (O + I) * r`` per module instead of
    ``n_queries * O * I``.
    """
    factors: dict[str, tuple[Tensor, Tensor]] = {}
    for m in modules:
        q = query_grads[m].to(device=device, dtype=torch.float32)
        O, I = grad_shapes[m]
        q_mat = q.reshape(-1, O, I)  # [nq, O, I]

        r = min(rank, O, I)
        U, S, Vh = torch.linalg.svd(q_mat, full_matrices=False)
        # Truncate to rank r
        U = U[:, :, :r]  # [nq, O, r]
        S = S[:, :r]  # [nq, r]
        Vh = Vh[:, :r, :]  # [nq, r, I]

        # Absorb singular values into U
        U = U * S.unsqueeze(1)  # [nq, O, r]
        V = Vh.transpose(-2, -1)  # [nq, I, r]

        factors[m] = (U.to(dtype), V.to(dtype))

    return factors


class Scorer:
    """
    Scores training gradients against query gradients.

    Accepts an optional ``index_transform`` callable that is applied to each
    batch of index gradients before scoring. This can be used for
    preconditioning, projection, or any other per-batch transformation.
    When no transform is needed, pass ``None`` (identity is used).

    Accepts a ScoreWriter for saving the scores (disk or in-memory).
    """

    def __init__(
        self,
        query_grads: dict[str, Tensor],
        modules: list[str],
        writer: ScoreWriter,
        device: torch.device,
        dtype: torch.dtype,
        *,
        unit_normalize: bool = False,
        score_mode: str = "individual",
        attribute_tokens: bool = False,
        index_transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] = lambda x: x,
        low_rank: int = 0,
        grad_shapes: dict[str, list[int]] | None = None,
    ):
        """
        Initialize the scorer.

        Parameters
        ----------
        query_grads : dict[str, Tensor]
            Query gradients keyed by module name. Should already be
            preconditioned if preconditioning is desired.
        modules : list[str]
            List of module names to use for scoring.
        writer : ScoreWriter
            Writer for score output (InMemoryScoreWriter or MemmapScoreWriter).
        device : torch.device
            Device to perform scoring on.
        dtype : torch.dtype
            Dtype for scoring computation.
        unit_normalize : bool
            Whether to unit normalize gradients before scoring.
        score_mode : str
            Scoring mode: "individual" or "nearest".
        attribute_tokens : bool
            Whether gradients are per-token (rows = total_valid tokens).
        index_transform : Callable | None
            Optional transform applied to index gradients per-batch before
            scoring. Receives and returns ``dict[str, Tensor]``. When ``None``,
            index gradients are used as-is.
        low_rank : int
            When > 0, compress query gradients to this rank via truncated SVD.
        grad_shapes : dict[str, list[int]] | None
            Per-module weight shapes ``[O, I]``.  Required when
            ``low_rank > 0``.
        """
        self.device = device
        self.dtype = dtype
        self.modules = modules
        self.unit_normalize = unit_normalize
        self.score_mode = score_mode
        self.attribute_tokens = attribute_tokens
        self.writer = writer
        self.index_transform = index_transform
        self.low_rank = low_rank
        self.grad_shapes = grad_shapes

        if low_rank > 0:
            assert grad_shapes is not None, (
                "grad_shapes are required for low-rank query compression. "
                "Rebuild the query index with a recent bergson version."
            )
            self.query_factors = _compute_low_rank_factors(
                query_grads,
                modules,
                grad_shapes,
                low_rank,
                device,
                dtype,
            )
            self.query_grads_t = None
        else:
            self.query_factors = None
            # Store per-module transposed query grads: {m: [dim_m, n_queries]}
            # Keeping them separate avoids materialising the full
            # [total_dim, n_queries] concatenation (which can be 40+ GiB with
            # ekfac).
            self.query_grads_t = {
                m: query_grads[m].to(device=self.device, dtype=self.dtype).T
                for m in modules
            }

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, Tensor],
    ):
        """Score a batch of training gradients against all queries."""
        scores = self.score(mod_grads)
        self.writer(indices, scores)

    @torch.inference_mode()
    def score(self, index_grads: dict[str, Tensor]) -> Tensor:
        """Compute scores for a batch of gradients."""
        index_grads = self.index_transform(index_grads)

        # Accumulate scores and (optionally) squared norms module-by-module to
        # avoid concatenating all gradients into one giant GPU tensor.
        scores: Tensor | None = None
        norm_sq: Tensor | None = None

        for m in self.modules:
            idx = index_grads[m].to(self.device, self.dtype, non_blocking=True)

            if self.low_rank > 0:
                assert self.query_factors is not None
                contrib = self._score_low_rank(idx, m)
            else:
                assert self.query_grads_t is not None
                contrib = idx @ self.query_grads_t[m]

            scores = contrib if scores is None else scores.add_(contrib)

            if self.unit_normalize:
                sq = idx.pow(2).sum(dim=1)
                norm_sq = sq if norm_sq is None else norm_sq.add_(sq)

        assert scores is not None

        if self.unit_normalize:
            assert norm_sq is not None
            i_norm = norm_sq.sqrt_().clamp_min_(1e-12).unsqueeze_(1)
            scores.div_(i_norm)

        if self.score_mode == "nearest":
            return scores.max(dim=-1).values

        return scores

    def _score_low_rank(self, idx: Tensor, module: str) -> Tensor:
        """Score a batch of flat training gradients against low-rank query factors.

        For module *m* with weight shape (O, I):
          idx:  [batch, O*I]  (flat training gradient)
          U_m:  [nq, O, r]   (left factors, singular values absorbed)
          V_m:  [nq, I, r]   (right factors)

        The score is ``trace(G_n^T @ U_q @ V_q^T)`` for each (n, q) pair.
        We compute this as ``sum((G_n @ V_q) * U_q)`` looping over small
        groups of queries to bound peak memory.
        """
        assert self.query_factors is not None and self.grad_shapes is not None
        U, V = self.query_factors[module]
        O, I = self.grad_shapes[module]
        nq = U.shape[0]
        batch = idx.shape[0]
        r = U.shape[2]

        idx_mat = idx.reshape(batch, O, I)  # [batch, O, I]

        # Process queries in chunks to bound intermediate memory at
        # batch * chunk * O * r elements.
        chunk = max(1, min(nq, (64 * 1024 * 1024) // max(1, batch * O * r)))
        out = idx.new_zeros(batch, nq)

        for q0 in range(0, nq, chunk):
            q1 = min(q0 + chunk, nq)
            V_c = V[q0:q1]  # [c, I, r]
            U_c = U[q0:q1]  # [c, O, r]
            # [batch, O, I] @ [c, r, I]^T → einsum is clearest
            # A[b, c, O, r] = sum_I idx_mat[b, O, I] * V_c[c, I, r]
            A = torch.einsum("boi,cir->bcor", idx_mat, V_c)
            # scores[b, c] = sum_{O,r} A[b, c, O, r] * U_c[c, O, r]
            out[:, q0:q1] = torch.einsum("bcor,cor->bc", A, U_c)

        return out
