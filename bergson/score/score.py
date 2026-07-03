import json
import os
import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, IterableDataset
from tqdm.auto import tqdm

from bergson.collection import collect_gradients
from bergson.config import IndexConfig, PreprocessConfig, ScoreConfig
from bergson.data import (
    allocate_batches,
    load_gradients,
)
from bergson.distributed import launch_distributed_run
from bergson.process_grads import (
    get_trackstar_preconditioner,
    normalize_and_aggregate_grads,
)
from bergson.score.score_writer import (
    MemmapSequenceScoreWriter,
    MemmapTokenScoreWriter,
    ScoreWriter,
)
from bergson.score.scorer import Scorer
from bergson.utils.utils import (
    assert_type,
    convert_precision_to_torch,
    get_gradient_dtype,
)
from bergson.utils.worker_utils import (
    create_processor,
    setup_data_pipeline,
    setup_model_and_peft,
)


def _peek_n_queries(query_path: str) -> int:
    """Return the number of queries in a gradient index without loading tensors."""
    with open(Path(query_path) / "info.json") as f:
        return int(json.load(f)["num_grads"])


def get_query_grads(
    score_cfg: ScoreConfig,
    row_range: tuple[int, int] | None = None,
) -> tuple[dict[str, torch.Tensor], PreprocessConfig, dict[str, list[int]] | None]:
    """
    Load query gradients from the mmap index and return as a dict of tensors.

    Parameters
    ----------
    score_cfg : ScoreConfig
        Score configuration specifying the query path and target modules.
    row_range : tuple[int, int] | None
        If given, load only rows ``[row_range[0], row_range[1])`` from the
        query index.  Useful for chunked scoring to keep memory bounded.

    Returns
    -------
    tuple[dict[str, torch.Tensor], PreprocessConfig, dict[str, list[int]] | None]
        The query gradients, any preprocessing config embedded in the index,
        and optionally the per-module weight shapes ``[O, I]`` (``None`` for
        older indices that lack this metadata).
    """
    query_path = Path(score_cfg.query_path)
    if not query_path.exists():
        raise FileNotFoundError(
            f"Query dataset not found at {score_cfg.query_path}. "
            "Please build a query dataset index first."
        )

    with open(query_path / "info.json", "r") as f:
        metadata = json.load(f)
        target_modules = metadata["dtype"]["names"]
        grad_sizes = metadata["grad_sizes"]
        grad_shapes = metadata.get("grad_shapes")

    preprocess_path = Path(query_path / "preprocess_config.yaml")
    if preprocess_path.exists():
        preprocess_cfg = PreprocessConfig.load(preprocess_path)
    else:
        preprocess_cfg = PreprocessConfig()

    if not score_cfg.modules:
        score_cfg.modules = target_modules

    mmap = load_gradients(Path(score_cfg.query_path), structured=False)

    sizes = torch.tensor(list(grad_sizes.values()))
    module_offsets = torch.tensor([0] + torch.cumsum(sizes, dim=0).tolist())

    # Cast to float32 only for dtypes not natively supported by numpy (e.g. bfloat16)
    needs_cast = not np.issubdtype(mmap.dtype, np.floating)
    grads: dict[str, torch.Tensor] = {}
    for i, name in enumerate(grad_sizes.keys()):
        if name not in target_modules:
            continue
        if row_range is not None:
            sliced = mmap[
                row_range[0] : row_range[1], module_offsets[i] : module_offsets[i + 1]
            ]
        else:
            sliced = mmap[:, module_offsets[i] : module_offsets[i + 1]]
        if needs_cast:
            grads[name] = torch.from_numpy(sliced.astype(np.float32))
        else:
            grads[name] = torch.from_numpy(sliced.copy())

    return grads, preprocess_cfg, grad_shapes


def _make_split_preconditioner(
    preconditioners: dict[str, torch.Tensor],
    modules: list[str],
    device: torch.device,
    dtype: torch.dtype,
):
    """Build a per-batch index transform for split (two-sided) preconditioning."""
    stacked = torch.stack([preconditioners[m] for m in modules])

    def transform(
        grads: dict[str, torch.Tensor],
        _modules: list[str] = modules,
        _stacked: torch.Tensor = stacked,
        _device: torch.device = device,
        _dtype: torch.dtype = dtype,
    ) -> dict[str, torch.Tensor]:
        g = torch.stack(
            [grads[m].to(_device, _dtype, non_blocking=True) for m in _modules],
            dim=1,
        )
        result = torch.bmm(g.permute(1, 0, 2), _stacked).permute(1, 0, 2)
        return {m: result[:, i] for i, m in enumerate(_modules)}

    return transform


def create_scorer(
    path: Path,
    data: Dataset,
    score_cfg: ScoreConfig,
    preprocess_cfg: PreprocessConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    attribute_tokens: bool = False,
    row_range: tuple[int, int] | None = None,
    column_offset: int = 0,
    total_num_scores: int | None = None,
    writer: ScoreWriter | None = None,
) -> Scorer:
    """Create a Scorer with MemmapScoreWriter for disk-based scoring.

    Loads query gradients from disk, preprocesses them if not already
    preprocessed, and constructs the Scorer.

    * Loads preconditioner from ``preprocess_cfg.preconditioner_path``.
    * Applies to query grads once here (unless already preconditioned).
    * Normalizes and aggregates (unless already done).
    * Builds an ``index_transform`` closure for per-batch index
      preconditioning in split mode (``unit_normalize=True``).

    Parameters
    ----------
    row_range : tuple[int, int] | None
        If given, load only rows ``[row_range[0], row_range[1])`` from the
        query index.  Used by chunked scoring to keep memory bounded.
    column_offset : int
        Column index at which this scorer starts writing into the output
        memmap. Used for chunked scoring so each chunk writes directly into
        its column range of the final file.
    total_num_scores : int | None
        Total number of score columns in the output memmap. Defaults to the
        number of query grads loaded (i.e. the non-chunked case).
    writer : ScoreWriter | None
        If provided, use this writer instead of creating a
        MemmapSequenceScoreWriter. path, data, attribute_tokens,
        column_offset, and total_num_scores are ignored.
    """
    query_grads, query_preprocess_cfg, grad_shapes = get_query_grads(
        score_cfg, row_range=row_range
    )

    # Load preconditioner: H^(-1/2) for split, H^(-1) for one-sided
    preconditioners = get_trackstar_preconditioner(
        preprocess_cfg.preconditioner_path,
        device=device,
        power=-0.5 if preprocess_cfg.unit_normalize else -1,
        return_dtype=dtype,
    )

    # Maybe precondition query grads if it hasn't already been applied, e.g.
    # during reduce.
    if preconditioners and not bool(query_preprocess_cfg.preconditioner_path):
        query_grads = {
            m: query_grads[m].to(device=device, dtype=dtype) @ preconditioners[m]
            for m in score_cfg.modules
        }

    # Build index_transform for split (two-sided) preconditioning
    index_transform = (
        _make_split_preconditioner(
            preconditioners,
            score_cfg.modules,
            device,
            dtype,
        )
        if preconditioners and preprocess_cfg.unit_normalize
        else lambda x: x
    )

    # Maybe apply aggregation if it hasn't already been applied.
    normalize_aggregated_grad = (
        False
        if query_preprocess_cfg.normalize_aggregated_grad
        else preprocess_cfg.normalize_aggregated_grad
    )
    aggregation = (
        "none"
        if query_preprocess_cfg.aggregation != "none"
        else preprocess_cfg.aggregation
    )
    unit_normalize = (
        False if query_preprocess_cfg.unit_normalize else preprocess_cfg.unit_normalize
    )

    query_grads = normalize_and_aggregate_grads(
        query_grads,
        score_cfg.modules,
        unit_normalize=unit_normalize,
        device=device,
        aggregate_grads=aggregation,
        normalize_aggregated_grad=normalize_aggregated_grad,
    )

    num_queries = len(query_grads[score_cfg.modules[0]])
    if writer is None:
        if attribute_tokens:
            writer = MemmapTokenScoreWriter(
                path,
                data,
                num_queries,
                dtype=dtype,
                column_offset=column_offset,
                total_num_scores=total_num_scores,
            )
        else:
            writer = MemmapSequenceScoreWriter(
                path,
                len(data),
                num_queries,
                dtype=dtype,
                column_offset=column_offset,
                total_num_scores=total_num_scores,
            )

    return Scorer(
        query_grads=query_grads,
        modules=score_cfg.modules,
        writer=writer,
        device=device,
        dtype=dtype,
        unit_normalize=preprocess_cfg.unit_normalize,
        score_mode=score_cfg.score,
        attribute_tokens=attribute_tokens,
        index_transform=index_transform,
        low_rank=score_cfg.query_low_rank,
        grad_shapes=grad_shapes,
    )


def score_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    index_cfg: IndexConfig,
    score_cfg: ScoreConfig,
    preprocess_cfg: PreprocessConfig,
    ds: Dataset | IterableDataset,
):
    """
    Score worker executed per rank to produce and score gradients against a query.

    Parameters
    ----------
    rank : int
        Distributed rank / GPU ID for this worker.
    local_rank : int
        Local rank / GPU ID for this worker on the node.
    world_size : int
        Total number of workers participating in the run.
    index_cfg : IndexConfig
        Specifies the model, tokenizer, PEFT adapters, and other settings.
    score_cfg : ScoreConfig
        Score configuration specifying query path, target modules, and scoring
        method (mean/nearest/individual).
    preprocess_cfg : PreprocessConfig
        Preprocessing configuration for gradient normalization/preconditioning.
    ds : Dataset | IterableDataset
        The entire dataset to be indexed. A subset is assigned to each worker.
    """
    torch.cuda.set_device(local_rank)

    # These should be set by the main process
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

    model, target_modules = setup_model_and_peft(index_cfg)
    processor = create_processor(model, index_cfg, target_modules)

    attention_cfgs = {
        module: index_cfg.attention for module in index_cfg.split_attention_modules
    }

    kwargs = {
        "model": model,
        "data": ds,
        "processor": processor,
        "cfg": index_cfg,
        "target_modules": target_modules,
        "attention_cfgs": attention_cfgs,
    }

    score_dtype = (
        convert_precision_to_torch(score_cfg.precision)
        if score_cfg.precision != "auto"
        else get_gradient_dtype(model)
    )
    score_device = torch.device(f"cuda:{rank}")

    if isinstance(ds, Dataset):
        kwargs["batches"] = allocate_batches(
            ds["length"][:],
            index_cfg.token_batch_size,
            max_batch_size=index_cfg.max_batch_size,
        )

        n_queries = _peek_n_queries(score_cfg.query_path)
        chunk_size = score_cfg.query_chunk_size

        if chunk_size > 0 and n_queries > chunk_size:
            # Chunked scoring: run one pass per query chunk to bound peak RAM.
            # The model is evaluated ceil(n_queries / chunk_size) times. Each
            # chunk writes directly into its column range of the final memmap
            # — no intermediate files, no merge step.
            chunks = [
                (q, min(q + chunk_size, n_queries))
                for q in range(0, n_queries, chunk_size)
            ]
            for q_start, q_end in chunks:
                kwargs["scorer"] = create_scorer(
                    index_cfg.partial_run_path,
                    ds,
                    score_cfg,
                    preprocess_cfg,
                    device=score_device,
                    dtype=score_dtype,
                    attribute_tokens=index_cfg.attribute_tokens,
                    row_range=(q_start, q_end),
                    column_offset=q_start,
                    total_num_scores=n_queries,
                )
                collect_gradients(**kwargs)
                # Flush the chunk's writes before the scorer is replaced, so
                # data is durably on disk even if a later chunk fails.
                kwargs["scorer"].writer.flush()
        else:
            kwargs["scorer"] = create_scorer(
                index_cfg.partial_run_path,
                ds,
                score_cfg,
                preprocess_cfg,
                device=score_device,
                dtype=score_dtype,
                attribute_tokens=index_cfg.attribute_tokens,
            )
            collect_gradients(**kwargs)
    else:
        # Convert each shard to a Dataset then map over its gradients
        buf, shard_id = [], 0

        def flush(kwargs):
            nonlocal buf, shard_id
            if not buf:
                return
            ds_shard = assert_type(Dataset, Dataset.from_list(buf))
            batches = allocate_batches(
                ds_shard["length"][:],
                index_cfg.token_batch_size,
                max_batch_size=index_cfg.max_batch_size,
            )
            kwargs["ds"] = ds_shard
            kwargs["batches"] = batches

            kwargs["scorer"] = create_scorer(
                index_cfg.partial_run_path / f"shard-{shard_id:05d}",
                ds_shard,
                score_cfg,
                preprocess_cfg,
                device=score_device,
                dtype=score_dtype,
            )

            collect_gradients(**kwargs)

            buf.clear()
            shard_id += 1

        for ex in tqdm(ds, desc="Collecting gradients"):
            buf.append(ex)
            if len(buf) == index_cfg.stream_shard_size:
                flush(kwargs=kwargs)

        flush(kwargs=kwargs)  # Final flush
        if rank == 0:
            processor.save(index_cfg.partial_run_path)


def score_dataset(
    index_cfg: IndexConfig,
    score_cfg: ScoreConfig,
    preprocess_cfg: PreprocessConfig,
):
    """
    Score a dataset against an existing gradient index.

    Parameters
    ----------
    index_cfg : IndexConfig
        Specifies the run path, dataset, model, tokenizer, PEFT adapters,
        and other gradient collection settings.
    score_cfg : ScoreConfig
        Specifies the query path, target modules, and scoring method
        (mean/nearest/individual).
    preprocess_cfg : PreprocessConfig
        Preprocessing configuration for gradient normalization/preconditioning.
    """
    index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)

    index_cfg.save_yaml(index_cfg.partial_run_path / "index_config.yaml")
    score_cfg.save_yaml(index_cfg.partial_run_path / "score_config.yaml")

    ds, _ = setup_data_pipeline(index_cfg)

    launch_distributed_run(
        "score",
        score_worker,
        [index_cfg, score_cfg, preprocess_cfg, ds],
        index_cfg.distributed,
    )

    if index_cfg.distributed.rank == 0:
        shutil.move(index_cfg.partial_run_path, index_cfg.run_path)


def score_from_index(
    train_index_path: Path,
    score_cfg: ScoreConfig,
    preprocess_cfg: PreprocessConfig,
    out_path: Path,
    *,
    train_chunk_size: int = 2048,
    device: torch.device | None = None,
) -> None:
    """Score a prebuilt training gradient index against a query index.

    No model inference required. Loads training gradients from disk in chunks
    of ``train_chunk_size`` rows to bound peak GPU memory.

    The training index must already have any per-parameter normalisation (e.g.
    Adam second-moment correction for TrackStar) baked in at build time via
    ``IndexConfig.processor_path``.  Any Hessian preconditioner
    (``preprocess_cfg.preconditioner_path``) is applied on-the-fly to each
    chunk, identical to how ``score_dataset`` applies it.

    Parameters
    ----------
    train_index_path:
        Directory of the prebuilt training gradient index (contains
        ``gradients.bin`` and ``info.json``).
    score_cfg:
        Score config; ``query_path`` must point to an existing query index.
        ``modules`` is inferred from the training index if empty.
    preprocess_cfg:
        Preprocessing config (``unit_normalize``, ``preconditioner_path``).
        Applied identically to the model-based scoring path.
    out_path:
        Output directory.  Written atomically via a ``.part`` sibling.
    train_chunk_size:
        Number of training examples loaded into GPU memory per step.
    device:
        Target device.  Defaults to ``cuda:0`` if available, else ``cpu``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = convert_precision_to_torch(score_cfg.precision)

    # ── Load training index metadata ──────────────────────────────────────
    with open(train_index_path / "info.json") as f:
        train_info = json.load(f)
    train_grad_sizes: dict[str, int] = train_info["grad_sizes"]
    n_train: int = train_info["num_grads"]

    module_names = list(train_grad_sizes.keys())
    mod_sizes = list(train_grad_sizes.values())
    mod_offsets = [0] + list(np.cumsum(mod_sizes))

    if not score_cfg.modules:
        score_cfg.modules = module_names

    # Memory-mapped view of training gradients — no RAM cost until accessed.
    train_mmap = load_gradients(train_index_path, structured=False)
    needs_cast = not np.issubdtype(train_mmap.dtype, np.floating)

    # ── Set up query side (reuses create_scorer internals) ────────────────
    partial_out = Path(str(out_path) + ".part")
    partial_out.mkdir(parents=True, exist_ok=True)

    query_grads, query_preprocess_cfg, grad_shapes = get_query_grads(score_cfg)

    preconditioners = get_trackstar_preconditioner(
        preprocess_cfg.preconditioner_path,
        device=device,
        power=-0.5 if preprocess_cfg.unit_normalize else -1,
        return_dtype=dtype,
    )

    if preconditioners and not bool(query_preprocess_cfg.preconditioner_path):
        query_grads = {
            m: query_grads[m].to(device=device, dtype=dtype) @ preconditioners[m]
            for m in score_cfg.modules
        }

    index_transform = (
        _make_split_preconditioner(preconditioners, score_cfg.modules, device, dtype)
        if preconditioners and preprocess_cfg.unit_normalize
        else lambda x: x
    )

    unit_normalize = (
        False if query_preprocess_cfg.unit_normalize else preprocess_cfg.unit_normalize
    )
    aggregation = (
        "none"
        if query_preprocess_cfg.aggregation != "none"
        else preprocess_cfg.aggregation
    )
    normalize_aggregated_grad = (
        False
        if query_preprocess_cfg.normalize_aggregated_grad
        else preprocess_cfg.normalize_aggregated_grad
    )
    query_grads = normalize_and_aggregate_grads(
        query_grads,
        score_cfg.modules,
        unit_normalize=unit_normalize,
        device=device,
        aggregate_grads=aggregation,
        normalize_aggregated_grad=normalize_aggregated_grad,
    )

    n_queries = len(query_grads[score_cfg.modules[0]])
    writer = MemmapSequenceScoreWriter(partial_out, n_train, n_queries, dtype=dtype)

    scorer = Scorer(
        query_grads=query_grads,
        modules=score_cfg.modules,
        writer=writer,
        device=device,
        dtype=dtype,
        unit_normalize=preprocess_cfg.unit_normalize,
        score_mode=score_cfg.score,
        attribute_tokens=False,
        index_transform=index_transform,
        low_rank=score_cfg.query_low_rank,
        grad_shapes=grad_shapes,
    )

    # ── Iterate training chunks ───────────────────────────────────────────
    for start in tqdm(range(0, n_train, train_chunk_size), desc="Scoring from index"):
        end = min(start + train_chunk_size, n_train)
        chunk_np = train_mmap[start:end]
        if needs_cast:
            chunk_np = chunk_np.astype(np.float32)
        chunk_t = torch.from_numpy(chunk_np.copy()).to(device, dtype)

        mod_grads = {
            name: chunk_t[:, mod_offsets[i] : mod_offsets[i + 1]]
            for i, name in enumerate(module_names)
            if name in score_cfg.modules
        }
        scorer(list(range(start, end)), mod_grads)

    writer.flush()
    shutil.move(str(partial_out), str(out_path))


def score_dataset_streaming(
    index_cfg: IndexConfig,
    score_cfg: ScoreConfig,
    preprocess_cfg: PreprocessConfig,
    writer: ScoreWriter,
) -> None:
    """Score a dataset against a query index, writing results to a custom writer.

    Like score_dataset but bypasses all disk-based memmap output — scores are
    passed directly to *writer* batch by batch. Single-GPU only (no distributed
    support). Use this to avoid writing large (N_train × 2m) score files.

    Parameters
    ----------
    index_cfg : IndexConfig
        Model, tokenizer, and data settings. ``run_path`` is ignored.
    score_cfg : ScoreConfig
        Query index path and scoring mode.
    preprocess_cfg : PreprocessConfig
        Normalization / preconditioner settings.
    writer : ScoreWriter
        Receives ``(indices, scores)`` for each gradient batch.
    """
    ds, _ = setup_data_pipeline(index_cfg)
    if not isinstance(ds, Dataset):
        raise TypeError(
            "score_dataset_streaming requires a Dataset, not IterableDataset"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, target_modules = setup_model_and_peft(index_cfg)
    score_dtype = (
        convert_precision_to_torch(score_cfg.precision)
        if score_cfg.precision != "auto"
        else get_gradient_dtype(model)
    )
    processor = create_processor(model, index_cfg, target_modules)
    attention_cfgs = {
        module: index_cfg.attention for module in index_cfg.split_attention_modules
    }
    batches = allocate_batches(
        ds["length"][:], index_cfg.token_batch_size, max_batch_size=index_cfg.max_batch_size
    )

    scorer = create_scorer(
        Path("."),  # unused — writer is provided
        ds,
        score_cfg,
        preprocess_cfg,
        device=device,
        dtype=score_dtype,
        attribute_tokens=index_cfg.attribute_tokens,
        writer=writer,
    )

    collect_gradients(
        model=model,
        data=ds,
        processor=processor,
        cfg=index_cfg,
        target_modules=target_modules,
        attention_cfgs=attention_cfgs,
        batches=batches,
        scorer=scorer,
    )
    writer.flush()
