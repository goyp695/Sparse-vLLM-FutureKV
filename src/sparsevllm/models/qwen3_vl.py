from __future__ import annotations

import torch
from torch import nn
import torch.distributed as dist

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from sparsevllm.layers.activation import SiluAndMul
from sparsevllm.layers.attention import Attention
from sparsevllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from sparsevllm.layers.layernorm import RMSNorm
from sparsevllm.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from sparsevllm.layers.rotary_embedding import apply_rotary_emb
from sparsevllm.utils.context import get_context


def _get_text_config(config):
    return getattr(config, "text_config", config)


def _get_rope_theta(config) -> float:
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        return rope_parameters["rope_theta"]
    if hasattr(config, "rope_theta"):
        return config.rope_theta
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and "rope_theta" in rope_scaling:
        return rope_scaling["rope_theta"]
    return 10000.0


def _get_mrope_section(config) -> list[int]:
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "mrope_section" in rope_parameters:
        return list(rope_parameters["mrope_section"])
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and "mrope_section" in rope_scaling:
        return list(rope_scaling["mrope_section"])
    return [24, 20, 20]


class Qwen3VLMRotaryEmbedding(nn.Module):
    """Qwen3-VL MRoPE for flattened Sparse-vLLM token batches."""

    def __init__(
        self,
        head_dim: int,
        max_position: int,
        base: float,
        mrope_section: list[int],
    ) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.max_position = int(max_position)
        self.mrope_section = [int(x) for x in mrope_section]
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        # freqs: [3, num_tokens, head_dim // 2]
        out = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            out[..., slice(offset, length, 3)] = freqs[dim, ..., slice(offset, length, 3)]
        return out

    def forward(
        self,
        mrope_positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mrope_positions.ndim == 1:
            mrope_positions = mrope_positions.unsqueeze(0).expand(3, -1)
        if mrope_positions.ndim != 2 or int(mrope_positions.shape[0]) != 3:
            raise ValueError(f"Qwen3-VL MRoPE expects [3, N] or [N] positions, got {tuple(mrope_positions.shape)}.")

        pos = mrope_positions.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = pos[:, :, None] * self.inv_freq[None, None, :]
        freqs = self._apply_interleaved_mrope(freqs.float())
        cos = freqs.cos().unsqueeze(1).to(dtype=query.dtype)
        sin = freqs.sin().unsqueeze(1).to(dtype=query.dtype)
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)


class Qwen3VLTextAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        head_dim: int,
        rms_norm_eps: float,
        qkv_bias: bool,
        rope_theta: float,
        mrope_section: list[int],
        proj_chunk_size: int = 16384,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = int(head_dim or hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.proj_chunk_size = int(proj_chunk_size)
        if self.proj_chunk_size <= 0:
            raise ValueError(f"proj_chunk_size must be > 0, got {proj_chunk_size}.")

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=qkv_bias,
        )
        self.rotary_emb = Qwen3VLMRotaryEmbedding(
            self.head_dim,
            max_position=max_position,
            base=rope_theta,
            mrope_section=mrope_section,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def _o_proj_chunked(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        chunk_size = int(self.proj_chunk_size)
        if int(x.shape[0]) <= chunk_size:
            out.copy_(self.o_proj(x))
            return out
        for start in range(0, int(x.shape[0]), chunk_size):
            end = min(start + chunk_size, int(x.shape[0]))
            out[start:end].copy_(self.o_proj(x[start:end]))
        return out

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        context = get_context()
        cache_manager = context.cache_manager
        layer_idx = context.now_layer_idx
        cache_manager.save_raw_kv_if_needed(layer_idx, k, v)
        mrope_positions = getattr(context, "qwen3vl_mrope_positions", None)
        if mrope_positions is None:
            mrope_positions = positions
        q, k = self.rotary_emb(mrope_positions, q, k)
        cache_manager.save_rope_kv_if_needed(layer_idx, k, v)
        o = self.attn(q, k, v, positions=positions)
        return self._o_proj_chunked(o.flatten(1, -1), hidden_states)


class Qwen3VLTextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mlp_chunk_size: int = 16384,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()
        self.mlp_chunk_size = int(mlp_chunk_size)
        if self.mlp_chunk_size <= 0:
            raise ValueError(f"mlp_chunk_size must be > 0, got {mlp_chunk_size}.")

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        return self.down_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunk_size = int(self.mlp_chunk_size)
        if int(x.shape[0]) <= chunk_size:
            return self._forward_chunk(x)
        out = torch.empty_like(x)
        for start in range(0, int(x.shape[0]), chunk_size):
            end = min(start + chunk_size, int(x.shape[0]))
            out[start:end].copy_(self._forward_chunk(x[start:end]))
        return out


class Qwen3VLTextDecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.self_attn = Qwen3VLTextAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=config.attention_bias,
            head_dim=config.head_dim,
            rope_theta=_get_rope_theta(config),
            mrope_section=_get_mrope_section(config),
            proj_chunk_size=getattr(config, "mlp_chunk_size", 16384),
        )
        self.mlp = Qwen3VLTextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            mlp_chunk_size=getattr(config, "mlp_chunk_size", 16384),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


def _compute_qwen3vl_rope_index(
    input_ids: list[int],
    image_grid_thw: torch.Tensor | None,
    *,
    image_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, int]:
    seq_len = len(input_ids)
    if image_grid_thw is None or int((torch.tensor(input_ids) == image_token_id).sum().item()) == 0:
        position_ids = torch.arange(seq_len, dtype=torch.long).view(1, -1).expand(3, -1).clone()
        return position_ids, 0

    grid = image_grid_thw.to("cpu", dtype=torch.long)
    vision_start_indices = [idx for idx, token in enumerate(input_ids) if token == vision_start_token_id]
    image_nums = sum(
        1 for idx in vision_start_indices
        if idx + 1 < len(input_ids) and input_ids[idx + 1] == image_token_id
    )
    if image_nums != int(grid.shape[0]):
        raise ValueError(
            "Qwen3-VL image grid count does not match prompt image placeholders: "
            f"vision_start_images={image_nums} image_grid_thw={int(grid.shape[0])}."
        )

    input_tokens = list(input_ids)
    llm_pos_ids_list: list[torch.Tensor] = []
    st = 0
    image_index = 0
    for _ in range(image_nums):
        try:
            ed = input_tokens.index(image_token_id, st)
        except ValueError as exc:
            raise ValueError("Could not locate Qwen3-VL image placeholder token while building MRoPE ids.") from exc
        t, h, w = grid[image_index]
        image_index += 1
        llm_grid_t = int(t.item())
        llm_grid_h = int(h.item()) // int(spatial_merge_size)
        llm_grid_w = int(w.item()) // int(spatial_merge_size)
        text_len = ed - st
        st_idx = int(llm_pos_ids_list[-1].max().item()) + 1 if llm_pos_ids_list else 0
        if text_len > 0:
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    if st < seq_len:
        st_idx = int(llm_pos_ids_list[-1].max().item()) + 1 if llm_pos_ids_list else 0
        text_len = seq_len - st
        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

    position_ids = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    if int(position_ids.shape[1]) != seq_len:
        raise ValueError(
            f"Qwen3-VL MRoPE length mismatch: built={int(position_ids.shape[1])} prompt_len={seq_len}."
        )
    rope_delta = int(position_ids.max().item()) + 1 - seq_len
    return position_ids.to(dtype=torch.long), rope_delta


class Qwen3VLTextModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3VLTextDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sparse_controller = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        residual = None
        context = get_context()
        visual_pos_mask = getattr(context, "qwen3vl_visual_pos_mask", None)
        deepstack_embeds = getattr(context, "qwen3vl_deepstack_embeds", None)

        for i, layer in enumerate(self.layers):
            context.now_layer_idx = i
            hidden_states, residual = layer(positions, hidden_states, residual)
            if deepstack_embeds is not None and i < len(deepstack_embeds):
                visual_embeds = deepstack_embeds[i]
                if visual_pos_mask is not None and visual_embeds is not None and int(visual_embeds.numel()) > 0:
                    hidden_states = hidden_states.clone()
                    hidden_states[visual_pos_mask, :] = hidden_states[visual_pos_mask, :] + visual_embeds.to(
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
            if self.sparse_controller is not None:
                self.sparse_controller.on_layer_end(i, context)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3VLModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        text_config = _get_text_config(config)
        vision_config = config.vision_config
        if not hasattr(vision_config, "_attn_implementation"):
            vision_config._attn_implementation = "eager"
        else:
            vision_config._attn_implementation = "eager"
        self.visual = Qwen3VLVisionModel(vision_config)
        self.language_model = Qwen3VLTextModel(text_config)
        self.spatial_merge_size = int(vision_config.spatial_merge_size)
        self.image_token_id = int(config.image_token_id)
        self.vision_start_token_id = int(config.vision_start_token_id)
        self._seq_caches: dict[int, dict] = {}

    def free_seq(self, seq_id: int) -> None:
        self._seq_caches.pop(int(seq_id), None)

    def _ensure_seq_cache(self, seq):
        mm_data = getattr(seq, "multi_modal_data", None)
        if not mm_data:
            return None
        cache = self._seq_caches.get(int(seq.seq_id))
        if cache is None:
            cache = getattr(seq, "qwen3vl_cache", None)
        if cache is not None:
            self._seq_caches[int(seq.seq_id)] = cache
            return cache

        pixel_values = mm_data.get("pixel_values")
        image_grid_thw = mm_data.get("image_grid_thw")
        if pixel_values is None or image_grid_thw is None:
            return None

        if not torch.is_tensor(pixel_values):
            pixel_values = torch.as_tensor(pixel_values)
        if not torch.is_tensor(image_grid_thw):
            image_grid_thw = torch.as_tensor(image_grid_thw)
        visual_parameter = next(self.visual.parameters())
        visual_device = visual_parameter.device
        visual_dtype = visual_parameter.dtype
        image_grid_thw = image_grid_thw.to(device=visual_device, dtype=torch.long)
        pixel_values = pixel_values.to(device=visual_device, dtype=visual_dtype)

        with torch.inference_mode():
            image_embeds, deepstack_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

        visual_positions = [
            idx for idx, token_id in enumerate(seq.prompt_token_ids)
            if int(token_id) == self.image_token_id
        ]
        if len(visual_positions) != int(image_embeds.shape[0]):
            raise ValueError(
                "Qwen3-VL image token/features mismatch in Sparse-vLLM engine: "
                f"image_tokens={len(visual_positions)} image_features={int(image_embeds.shape[0])}."
            )
        visual_index_by_pos = torch.full((seq.num_prompt_tokens,), -1, dtype=torch.long)
        if visual_positions:
            visual_index_by_pos[torch.tensor(visual_positions, dtype=torch.long)] = torch.arange(
                len(visual_positions),
                dtype=torch.long,
            )

        mrope_position_ids, rope_delta = _compute_qwen3vl_rope_index(
            seq.prompt_token_ids,
            image_grid_thw.detach().cpu(),
            image_token_id=self.image_token_id,
            vision_start_token_id=self.vision_start_token_id,
            spatial_merge_size=self.spatial_merge_size,
        )
        cache = {
            "image_embeds": image_embeds,
            "deepstack_embeds": deepstack_embeds,
            "visual_index_by_pos": visual_index_by_pos,
            "mrope_position_ids": mrope_position_ids,
            "rope_delta": int(rope_delta),
        }
        seq.qwen3vl_cache = cache
        self._seq_caches[int(seq.seq_id)] = cache
        return cache

    def _prepare_sparsevllm_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        seqs = getattr(context, "seqs", None)
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        mrope_positions = positions.view(1, -1).expand(3, -1).clone()
        visual_pos_mask = torch.zeros((input_ids.numel(),), dtype=torch.bool, device=input_ids.device)
        deepstack_chunks: list[list[torch.Tensor]] | None = None

        if not seqs:
            context.qwen3vl_mrope_positions = mrope_positions
            context.qwen3vl_visual_pos_mask = visual_pos_mask
            context.qwen3vl_deepstack_embeds = None
            return inputs_embeds

        if context.cu_seqlens_q is not None and context.cu_seqlens_q.numel() > 1:
            spans = [
                (int(context.cu_seqlens_q[i].item()), int(context.cu_seqlens_q[i + 1].item()))
                for i in range(len(seqs))
            ]
        else:
            spans = [(i, i + 1) for i in range(len(seqs))]

        for seq, (start, end) in zip(seqs, spans):
            cache = self._ensure_seq_cache(seq)
            pos_b = positions[start:end].to(dtype=torch.long)
            if cache is None:
                continue

            prompt_len = int(cache["visual_index_by_pos"].numel())
            prompt_mask = pos_b < prompt_len
            if prompt_mask.any():
                prompt_flat_indices = torch.arange(start, end, device=input_ids.device, dtype=torch.long)[prompt_mask]
                prompt_pos = pos_b[prompt_mask].detach().cpu()
                mrope = cache["mrope_position_ids"].index_select(1, prompt_pos).to(device=input_ids.device)
                mrope_positions[:, prompt_flat_indices] = mrope

                visual_indices = cache["visual_index_by_pos"].index_select(0, prompt_pos).to(device=input_ids.device)
                has_visual = visual_indices >= 0
                if has_visual.any():
                    flat_indices = prompt_flat_indices[has_visual]
                    selected_visual_indices = visual_indices[has_visual].to(dtype=torch.long)
                    inputs_embeds[flat_indices, :] = cache["image_embeds"].index_select(
                        0,
                        selected_visual_indices,
                    ).to(device=input_ids.device, dtype=inputs_embeds.dtype)
                    visual_pos_mask[flat_indices] = True

                    deepstack = cache["deepstack_embeds"]
                    if deepstack_chunks is None:
                        deepstack_chunks = [[] for _ in range(len(deepstack))]
                    for idx, embed in enumerate(deepstack):
                        deepstack_chunks[idx].append(
                            embed.index_select(0, selected_visual_indices).to(
                                device=input_ids.device,
                                dtype=inputs_embeds.dtype,
                            )
                        )

            decode_mask = ~prompt_mask
            if decode_mask.any():
                delta = int(cache["rope_delta"])
                decode_pos = pos_b[decode_mask].to(device=input_ids.device) + delta
                decode_flat_indices = torch.arange(start, end, device=input_ids.device, dtype=torch.long)[decode_mask]
                mrope_positions[:, decode_flat_indices] = decode_pos.view(1, -1).expand(3, -1)

        context.qwen3vl_mrope_positions = mrope_positions
        context.qwen3vl_visual_pos_mask = visual_pos_mask
        if deepstack_chunks is None:
            context.qwen3vl_deepstack_embeds = None
        else:
            context.qwen3vl_deepstack_embeds = [
                torch.cat(chunks, dim=0) if chunks else None
                for chunks in deepstack_chunks
            ]
        return inputs_embeds

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self._prepare_sparsevllm_inputs(input_ids, positions)
        return self.language_model(input_ids, positions, inputs_embeds=inputs_embeds)


class Qwen3VLForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        text_config = _get_text_config(config)
        self.model = Qwen3VLModel(config)
        self.lm_head = ParallelLMHead(text_config.vocab_size, text_config.hidden_size)
        if text_config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.language_model.embed_tokens.weight.data

    @property
    def language_model(self):
        return self.model.language_model

    def free_seq(self, seq_id: int) -> None:
        self.model.free_seq(seq_id)

    @property
    def visual(self):
        return self.model.visual

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
