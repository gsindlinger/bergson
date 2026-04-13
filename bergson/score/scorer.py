from collections.abc import Callable

import torch
from torch import Tensor

from bergson.score.score_writer import ScoreWriter


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
        """
        self.device = device
        self.dtype = dtype
        self.modules = modules
        self.unit_normalize = unit_normalize
        self.score_mode = score_mode
        self.attribute_tokens = attribute_tokens
        self.writer = writer
        self.index_transform = index_transform

        # Store per-module transposed query grads: {m: [dim_m, n_queries]}
        # Keeping them separate avoids materialising the full [total_dim, n_queries]
        # concatenation (which can be 40+ GiB with ekfac).
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
