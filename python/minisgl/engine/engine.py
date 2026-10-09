from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.moe import create_moe_backend
from minisgl.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, _determine_cuda_graph_bs, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class Engine:
    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _adjust_config(config)
        self.is_qwen35 = config.model_config.model_type == "qwen3_5_text"
        self.eager_only = self.is_qwen35 or config.attention_backend == "quant-reference"

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))

        # ======================= KV cache initialization ========================
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
            max_running_req=config.max_running_req,
            kv_cache_dtype=config.kv_cache_dtype,
        )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = (
            None
            if self.is_qwen35
            else create_attention_backend(config.attention_backend, config.model_config)
        )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        self.graph_runner = None
        if self.is_qwen35:
            logger.info_rank0("Qwen3.5 uses eager execution and request-owned GDN states.")
            return
        if self.eager_only:
            logger.info_rank0("Quantization reference attention uses eager execution.")
            return

        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            if self.is_qwen35:
                return dict(load_weight(config.model_path, self.device))
            return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        num_cache_layers = (
            config.model_config.layer_types.count("full_attention")
            if self.is_qwen35
            else config.model_config.num_layers
        )
        fixed_state_memory = 0
        if self.is_qwen35:
            from minisgl.kvcache.hybrid_pool import gdn_state_bytes

            fixed_state_memory = gdn_state_bytes(
                config.model_config, config.max_running_req, self.dtype
            )
            logger.info_rank0(f"Reserving GDN request states: {mem_GB(fixed_state_memory)}")
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * num_cache_layers
        )
        self.kv_workspace_reserve = 0
        if config.kv_cache_dtype == "int8":
            from minisgl.kvcache.quant_pool import int8_kv_bytes_per_token

            heads = config.model_config.num_kv_heads
            dim = config.model_config.head_dim
            cache_per_page = int8_kv_bytes_per_token(num_cache_layers, heads, dim) * config.page_size
            if config.attention_backend == "quant-reference":
                # Reference reads gather INT8 data/scales and dequantize one
                # request/layer. SDPA/model temporaries use the remaining memory.
                self.kv_workspace_reserve = (
                    2 * heads * (dim * (1 + 4 + self.dtype.itemsize) + 4) * config.max_seq_len
                )
            else:
                # Split-K decode stores FP32 partial outputs and softmax stats.
                # Prefill dequantizes tiles on chip and has no history workspace.
                graph_sizes = _determine_cuda_graph_bs(
                    config.cuda_graph_bs, config.cuda_graph_max_bs, old_free_memory
                )
                max_bs = max(config.max_running_req, *(graph_sizes or [1]))
                self.kv_workspace_reserve = (
                    # Power-of-two buffers can retain graph pointers when an
                    # eager batch grows; their total capacity is below 2x max.
                    2 * (1 << (max_bs - 1).bit_length())
                    * config.model_config.num_qo_heads * 32 * (dim + 2) * 4
                )
                # Stable graph rows plus metadata snapshots for overlapping
                # batches. Raw token addresses use int32, not KV compute dtype.
                self.kv_workspace_reserve += (
                    4 * max_bs * _align_up_32(config.max_seq_len) * 4
                )
            fixed_state_memory += self.kv_workspace_reserve
            logger.info_rank0(
                f"Reserving INT8 attention workspace: {mem_GB(self.kv_workspace_reserve)}"
            )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = (
                int(config.memory_ratio * old_free_memory) - model_memory - fixed_state_memory
            )
            num_pages = available_memory // cache_per_page
            if config.kv_cache_dtype == "int8":
                num_pages -= 1  # the pool also owns a dummy page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        label = "INT8 K/V + FP32 scales" if config.kv_cache_dtype == "int8" else "K + V"
        logger.info(f"Allocating {num_tokens} tokens for KV cache, {label} = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            if self.graph_runner is not None and self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        if self.graph_runner is not None:
            self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    if config.kv_cache_dtype not in ("auto", "int8"):
        raise ValueError("kv_cache_dtype must be 'auto' or 'int8'")
    quant_backends = {"quant-reference", "quant-triton"}
    if len(config.attention_backend.split(",")) > 1 and quant_backends.intersection(
        config.attention_backend.split(",")
    ):
        raise NotImplementedError("quantized attention must be used as a standalone backend")
    if config.kv_cache_dtype == "int8":
        if (
            config.model_config.model_type != "qwen3"
            or config.tp_info.size != 1
            or config.dtype != torch.bfloat16
        ):
            raise NotImplementedError("INT8 KV currently supports Qwen3 Dense, TP=1, BF16 compute")
        if config.attention_backend not in ("auto", "quant-reference", "quant-triton"):
            raise ValueError("INT8 KV requires 'auto', 'quant-triton' or 'quant-reference'")
        if config.attention_backend == "auto":
            override("attention_backend", "quant-triton")
        if hasattr(config, "cache_type"):
            override("cache_type", "naive")

    if config.model_config.model_type == "qwen3_5_text":
        if config.tp_info.size != 1 or config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.5 Day 3 supports one GPU with BF16 weights")
        override("attention_backend", "eager")
        override("cuda_graph_bs", [])
        override("cuda_graph_max_bs", 0)
        if hasattr(config, "cache_type"):
            override("cache_type", "naive")
        return

    if config.attention_backend in quant_backends:
        if (
            config.model_config.model_type != "qwen3"
            or config.tp_info.size != 1
            or config.dtype != torch.bfloat16
        ):
            raise NotImplementedError("quantized attention supports Qwen3 Dense, TP=1, BF16")
        if config.attention_backend == "quant-triton" and config.kv_cache_dtype != "int8":
            raise ValueError("quant-triton requires kv_cache_dtype='int8'")
        if config.attention_backend == "quant-reference":
            override("cuda_graph_bs", [])
            override("cuda_graph_max_bs", 0)
        return

    if config.attention_backend == "auto":
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")
