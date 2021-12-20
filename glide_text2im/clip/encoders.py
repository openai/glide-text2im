import math
from collections import OrderedDict
from typing import List, Optional, Tuple, cast

import attr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (
    AttentionInfo,
    DenseAttentionMask,
    DenseCausalAttentionMask,
    make_full_layout,
    to_attention_info,
)
from .utils import Affine, LayerNorm, zero_key_bias_grad

# Constants used in the original CLIP implementation.
image_channel_means = [122.77093945, 116.74601272, 104.09373519]
image_channel_stds = [68.50053285, 66.63215831, 70.32316309]


@attr.s(eq=False, repr=False)
class TextEmbedding(nn.Module):
    n_vocab: int = attr.ib()
    n_context: int = attr.ib()
    n_state: int = attr.ib()
    device: torch.device = attr.ib(default=torch.device("cuda"))

    def __attrs_post_init__(self) -> None:
        super().__init__()

        w_voc = torch.empty((self.n_vocab, self.n_state), dtype=torch.float32, device=self.device)
        w_pos = torch.empty((self.n_context, self.n_state), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            w_voc.normal_(std=0.02)
            w_pos.normal_(std=0.01)

        self.w_voc = nn.Parameter(w_voc)
        self.w_pos = nn.Parameter(w_pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 2:
            raise ValueError()

        return F.embedding(x, self.w_voc) + self.w_pos[None, :, :]


@attr.s(eq=False, repr=False)
class ImageEmbedding(nn.Module):
    image_size: int = attr.ib()
    patch_size: int = attr.ib()
    n_state: int = attr.ib()
    n_timestep: int = attr.ib(default=0)
    device: torch.device = attr.ib(default=torch.device("cuda"))

    def __attrs_post_init__(self) -> None:
        super().__init__()

        if self.image_size % self.patch_size != 0:
            raise ValueError()

        n_patch = self.image_size // self.patch_size
        patch_proj = torch.empty(
            (self.n_state, 3) + 2 * (self.patch_size,), dtype=torch.float32, device=self.device
        )
        w_pos = torch.empty(
            (1 + n_patch ** 2, self.n_state), dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            if self.n_timestep == 0:
                pred_state = torch.empty((self.n_state,), dtype=torch.float32, device=self.device)
                pred_state.normal_(std=1 / np.sqrt(self.n_state))
                self.pred_state = nn.Parameter(pred_state)
            else:
                w_t = torch.empty(
                    (self.n_timestep, self.n_state), dtype=torch.float32, device=self.device
                )
                w_t.normal_(std=1 / np.sqrt(self.n_state))
                self.w_t = nn.Parameter(w_t)

            patch_proj.normal_(std=np.sqrt(2 / (self.n_state * self.patch_size ** 2)))
            w_pos.normal_(std=1 / np.sqrt(self.n_state))

        self.patch_proj = nn.Parameter(patch_proj)
        self.w_pos = nn.Parameter(w_pos)

        self.channel_means = torch.tensor(
            image_channel_means, dtype=torch.float32, device=self.device
        )[None, :, None, None]
        self.channel_stds = torch.tensor(
            image_channel_stds, dtype=torch.float32, device=self.device
        )[None, :, None, None]
        self.ln = LayerNorm(self.n_state, eps=1e-5, device=self.device)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if len(x.shape) != 4:
            raise ValueError("input should be 4d")
        if x.shape[1] != 3:
            raise ValueError("input should have 3 channels")
        if not (x.shape[2] == self.image_size and x.shape[3] == self.image_size):
            raise ValueError(f"input is not {self.image_size} x {self.image_size}")

        if (self.n_timestep == 0 and t is not None) or (self.n_timestep != 0 and t is None):
            raise ValueError()
        if self.n_timestep != 0:
            assert t is not None
            if len(t.shape) != 1:
                raise ValueError()
            if t.shape[0] != x.shape[0]:
                raise ValueError()

        x = (x - self.channel_means) / self.channel_stds
        x = F.conv2d(x, self.patch_proj, stride=self.patch_size)
        x = x.reshape(x.shape[0], self.n_state, (self.image_size // self.patch_size) ** 2).permute(
            0, 2, 1
        )

        sot = (
            self.pred_state[None, None].expand(x.shape[0], -1, -1)
            if self.n_timestep == 0
            else F.embedding(cast(torch.Tensor, t), self.w_t)[:, None]
        )
        x = torch.cat((sot, x), dim=1) + self.w_pos[None]
        return self.ln(x)


@attr.s(eq=False, repr=False)
class AttentionResblock(nn.Module):
    n_state: int = attr.ib()
    n_resblocks: int = attr.ib()
    attn_fn: AttentionInfo = attr.ib()
    device: torch.device = attr.ib(default=torch.device("cuda"))

    def __attrs_post_init__(self) -> None:
        super().__init__()

        self.n_head_state = self.n_state // self.attn_fn.n_heads
        self.qk_scale = 1 / np.sqrt(self.n_head_state)

        self.ln = LayerNorm(self.n_state, eps=1e-5, device=self.device)
        self.f_q = Affine(
            self.n_state,
            self.n_state,
            std=1 / math.sqrt(self.n_state),
            use_bias=True,
            bias_filter_fn=zero_key_bias_grad,
            device=self.device,
        )
        self.f_k = Affine(
            self.n_state,
            self.n_state,
            std=1 / math.sqrt(self.n_state),
            use_bias=False,
            bias_filter_fn=zero_key_bias_grad,
            device=self.device,
        )
        self.f_v = Affine(
            self.n_state,
            self.n_state,
            std=1 / math.sqrt(self.n_state),
            use_bias=True,
            bias_filter_fn=zero_key_bias_grad,
            device=self.device,
        )
        self.f_c = Affine(
            self.n_state,
            self.n_state,
            use_bias=True,
            std=1 / np.sqrt(self.n_state * self.n_resblocks ** 2),
            device=self.device,
        )  # XXX

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        n_context = m.shape[1]
        n_query_pad = self.attn_fn.ctx_blks_q * self.attn_fn.block_size - n_context
        n_key_pad = self.attn_fn.ctx_blks_k * self.attn_fn.block_size - n_context
        assert n_query_pad >= 0
        assert n_key_pad >= 0

        r = m
        r = self.ln(r)
        q, k, v = self.f_q(r), self.f_k(r), self.f_v(r)

        if n_query_pad != 0:
            q = F.pad(q, (0, 0, 0, n_query_pad))

        if n_key_pad != 0:
            k = F.pad(k, (0, 0, 0, n_key_pad))
            v = F.pad(v, (0, 0, 0, n_key_pad))

        q = q.view([q.shape[0], -1, self.attn_fn.n_heads, self.n_head_state]).permute((0, 2, 1, 3))
        k = k.view([k.shape[0], -1, self.attn_fn.n_heads, self.n_head_state]).permute((0, 2, 1, 3))
        v = v.view([v.shape[0], -1, self.attn_fn.n_heads, self.n_head_state]).permute((0, 2, 1, 3))
        w = torch.einsum(
            "bhcd,bhkd->bhck", q * math.sqrt(self.qk_scale), k * math.sqrt(self.qk_scale)
        )

        if hasattr(self.attn_fn, "pytorch_attn_bias"):
            bias = self.attn_fn.pytorch_attn_bias
            assert len(bias.shape) in {2, 3}

            if len(bias.shape) == 2:
                w = torch.softmax(w + self.attn_fn.pytorch_attn_bias[None, None], dim=-1)
            elif len(bias.shape) == 3:
                w = torch.softmax(w + self.attn_fn.pytorch_attn_bias[None], dim=-1)
        else:
            w = torch.softmax(w, dim=-1)

        r = torch.einsum("bhck,bhkd->bhcd", w, v)
        r = r.permute((0, 2, 1, 3)).reshape((r.shape[0], -1, self.n_state))

        if n_query_pad != 0:
            r = r[:, :-n_query_pad]

        assert r.shape[1] == n_context

        r = self.f_c(r)
        return m + r


@attr.s(eq=False, repr=False)
class FullyConnectedResblock(nn.Module):
    """
    Not imported from other files because we retain Alec's original inits.
    """

    n_state: int = attr.ib()
    n_resblocks: int = attr.ib()
    device: torch.device = attr.ib(default=torch.device("cuda"))

    def __attrs_post_init__(self) -> None:
        super().__init__()

        self.ln = LayerNorm(self.n_state, eps=1e-5, device=self.device)
        self.f_1 = Affine(
            self.n_state,
            4 * self.n_state,
            use_bias=True,
            std=np.sqrt(2 / (4 * self.n_state)),
            device=self.device,
        )
        self.f_2 = Affine(
            4 * self.n_state,
            self.n_state,
            use_bias=True,
            std=1 / np.sqrt(self.n_state * self.n_resblocks ** 2),
            device=self.device,
        )  # XXX

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        r = m
        r = self.ln(r)

        r = self.f_2(F.gelu(self.f_1(r)))
        return m + r


@attr.s(eq=False, repr=False)
class TransformerBlock(nn.Module):
    n_state: int = attr.ib()
    n_resblocks: int = attr.ib()
    attn_fn: AttentionInfo = attr.ib()
    device: torch.device = attr.ib(default=torch.device("cuda"))

    def __attrs_post_init__(self) -> None:
        super().__init__()

        self.f_attn = AttentionResblock(
            self.n_state,
            self.n_resblocks,
            self.attn_fn,
            self.device,
        )
        self.f_mlp = FullyConnectedResblock(self.n_state, self.n_resblocks, self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f_mlp(self.f_attn(x))


@attr.s(eq=False, repr=False)
class TextFeatureExtractor(nn.Module):
    n_state: int = attr.ib()
    n_embd: int = attr.ib()
    device: torch.device = attr.ib(default=torch.device("cuda"))

    def __attrs_post_init__(self) -> None:
        super().__init__()

        self.ln = LayerNorm(self.n_state, eps=1e-5, device=self.device)
        self.f = Affine(self.n_state, self.n_embd, use_bias=False, device=self.device)

    def forward(
        self, text: torch.Tensor, text_len: torch.Tensor, return_probe_features: bool = False
    ) -> torch.Tensor:
        if len(text.shape) != 3:
            raise ValueError("expected text to be 3d")
        if len(text_len.shape) != 1:
            raise ValueError("expected text length to be 1d")
        if text.shape[0] != text_len.shape[0]:
            raise ValueError("text and text_len have inconsistent batch dimensions")

        index = (text_len - 1)[:, None, None].expand(-1, 1, text.shape[2])
        x = torch.gather(text, dim=1, index=index)
        assert list(x.shape) == [text.shape[0], 1, text.shape[2]]

        if return_probe_features:
            return x[:, 0]

        x = self.ln(x)
        return self.f(x[:, 0])


@attr.s(eq=False, repr=False)
class ImageFeatureExtractor(nn.Module):
    n_state: int = attr.ib()
    n_embd: int = attr.ib()
    device: torch.device = attr.ib(default=torch.device("cuda"))

    def __attrs_post_init__(self) -> None:
        super().__init__()

        self.ln = LayerNorm(self.n_state, eps=1e-5, device=self.device)
        self.f = Affine(self.n_state, self.n_embd, use_bias=False, device=self.device)

    def forward(self, x: torch.Tensor, return_probe_features: bool = False) -> torch.Tensor:
        if return_probe_features:
            return x[:, 0]

        x = self.ln(x[:, :1])
        return self.f(x[:, 0])


@attr.s(eq=False, repr=False)
class TextEncoder(nn.Module):
    n_bpe_vocab: int = attr.ib()
    max_text_len: int = attr.ib()
    n_embd: int = attr.ib()
    n_head: int = attr.ib()
    n_xf_blocks: int = attr.ib()
    n_head_state: int = attr.ib(default=64)
    device: torch.device = attr.ib(default=torch.device("cuda"))
    block_size: int = attr.ib(init=False, default=32)

    def __attrs_post_init__(self) -> None:
        super().__init__()

        self.n_state = self.n_head * self.n_head_state
        n_rounded_context = self.block_size * int(math.ceil(self.max_text_len / self.block_size))
        n_pad = n_rounded_context - self.max_text_len

        args = (
            n_rounded_context,
            n_rounded_context,
            self.block_size,
            self.n_head,
            False,
            n_pad,
            n_pad,
        )
        mask = DenseCausalAttentionMask(*args)
        attn_fn = to_attention_info(mask)

        m = 1 - make_full_layout(mask).astype(np.float32)
        m[m == 1] = -1e10
        attn_fn.pytorch_attn_bias = torch.from_numpy(m).to(self.device)

        blocks: List[Tuple[str, nn.Module]] = [
            (
                "input",
                TextEmbedding(
                    self.n_bpe_vocab, self.max_text_len, self.n_state, device=self.device
                ),
            )
        ]

        for i in range(self.n_xf_blocks):
            blocks.append(
                (
                    f"block_{i}",
                    TransformerBlock(self.n_state, 2 * self.n_xf_blocks, attn_fn, self.device),
                )
            )

        blocks.append(
            ("output", TextFeatureExtractor(self.n_state, self.n_embd, device=self.device))
        )

        self.blocks = nn.ModuleDict(OrderedDict(blocks))

    def forward(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        return_probe_features: bool = False,
    ) -> torch.Tensor:

        n_batch = text.shape[0]
        h = self.blocks["input"](text)

        for i in range(self.n_xf_blocks):
            h = self.blocks[f"block_{i}"](h)

        h = self.blocks["output"](h, text_len, return_probe_features=return_probe_features)

        assert list(h.shape) == [
            n_batch,
            self.n_embd if not return_probe_features else self.n_state,
        ]
        return h


@attr.s(eq=False, repr=False)
class ImageEncoder(nn.Module):
    image_size: int = attr.ib()
    patch_size: int = attr.ib()
    n_embd: int = attr.ib()
    n_head: int = attr.ib()
    n_xf_blocks: int = attr.ib()
    n_head_state: int = attr.ib(default=64)
    n_timestep: int = attr.ib(default=0)
    device: torch.device = attr.ib(default=torch.device("cuda"))
    block_size: int = attr.ib(init=False, default=32)

    def __attrs_post_init__(self) -> None:
        super().__init__()

        self.n_state = self.n_head * self.n_head_state
        self.n_context = 1 + (self.image_size // self.patch_size) ** 2
        n_rounded_context = self.block_size * int(math.ceil(self.n_context / self.block_size))
        n_pad = n_rounded_context - self.n_context

        args = (
            n_rounded_context,
            n_rounded_context,
            self.block_size,
            self.n_head,
            False,
            n_pad,
            n_pad,
        )
        mask = DenseAttentionMask(*args)
        attn_fn = to_attention_info(mask)

        m = 1 - make_full_layout(mask).astype(np.float32)
        m[m == 1] = -1e10
        attn_fn.pytorch_attn_bias = torch.from_numpy(m).to(self.device)

        blocks: List[Tuple[str, nn.Module]] = [
            (
                "input",
                ImageEmbedding(
                    self.image_size,
                    self.patch_size,
                    self.n_state,
                    n_timestep=self.n_timestep,
                    device=self.device,
                ),
            )
        ]

        for i in range(self.n_xf_blocks):
            blocks.append(
                (
                    f"block_{i}",
                    TransformerBlock(self.n_state, 2 * self.n_xf_blocks, attn_fn, self.device),
                )
            )

        blocks.append(("output", ImageFeatureExtractor(self.n_state, self.n_embd, self.device)))

        self.blocks = nn.ModuleDict(OrderedDict(blocks))

    def forward(
        self,
        image: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        return_probe_features: bool = False,
    ) -> torch.Tensor:
        n_batch = image.shape[0]
        h = self.blocks["input"](image, t=timesteps)

        for i in range(self.n_xf_blocks):
            h = self.blocks[f"block_{i}"](h)

        h = self.blocks["output"](h, return_probe_features=return_probe_features)

        assert list(h.shape) == [
            n_batch,
            self.n_embd if not return_probe_features else self.n_state,
        ]

        return h
