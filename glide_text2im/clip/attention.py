import math
from abc import ABC, abstractmethod
from itertools import product
from typing import Any, Optional

import attr
import numpy as np
import torch


@attr.s
class AttentionMask(ABC):
    query_context_size: int = attr.ib(validator=lambda i, a, x: x >= 1)  # type: ignore
    key_context_size: int = attr.ib(validator=lambda i, a, x: x >= 1)  # type: ignore
    block_size: int = attr.ib(validator=lambda i, a, x: x >= 1)  # type: ignore
    n_head: int = attr.ib(validator=lambda i, a, x: x >= 1)  # type: ignore
    is_head_specific: bool = attr.ib(default=False)
    n_query_pad: int = attr.ib(default=0)
    n_key_pad: int = attr.ib(default=0)

    def __attrs_post_init__(self) -> None:
        if self.query_context_size % self.block_size != 0:
            raise ValueError()
        if self.key_context_size % self.block_size != 0:
            raise ValueError()
        if self.n_query_pad >= self.query_context_size:
            raise ValueError()
        if self.n_key_pad >= self.key_context_size:
            raise ValueError()

        self.n_query_block = self.query_context_size // self.block_size
        self.n_key_block = self.key_context_size // self.block_size
        self.first_pad_query_block_idx = self.n_query_block - int(
            math.ceil(self.n_query_pad / self.block_size)
        )
        self.first_pad_key_block_idx = self.n_key_block - int(
            math.ceil(self.n_key_pad / self.block_size)
        )

    def _make_global_layout(self) -> None:
        if not self.is_head_specific:
            m = np.ones([self.n_query_block, self.n_key_block], dtype=np.bool)
            r = product(*[range(n) for n in m.shape])

            for qb, kb in r:
                m[qb, kb] = np.any(self.block_layout(None, 0, qb, kb, 0))
        else:
            m = np.ones([self.n_head, self.n_query_block, self.n_key_block], dtype=np.bool)
            r = product(*[range(n) for n in m.shape])

            for h, qb, kb in r:
                m[h, qb, kb] = np.any(self.block_layout(None, h, qb, kb, 0))

        self.global_layout = m

    @abstractmethod
    def _block_layout(
        self, blk_shape: Any, head_idx: int, query_idx: int, key_idx: int, blk_idx: int
    ) -> np.ndarray:
        raise NotImplementedError()

    def block_layout(
        self, blk_shape: Any, head_idx: int, query_idx: int, key_idx: int, blk_idx: int
    ) -> np.ndarray:
        """
        `query_idx`, `key_idx` are block-level, zero-based indices.
        """

        m = np.ones([self.block_size, self.block_size], dtype=np.bool)

        if query_idx >= self.first_pad_query_block_idx:
            n_pad = min(
                self.block_size,
                (query_idx + 1) * self.block_size - (self.query_context_size - self.n_query_pad),
            )
            assert n_pad > 0
            m[self.block_size - n_pad :] = False
        if key_idx >= self.first_pad_key_block_idx:
            n_pad = min(
                self.block_size,
                (key_idx + 1) * self.block_size - (self.key_context_size - self.n_key_pad),
            )
            assert n_pad > 0
            m[:, self.block_size - n_pad :] = False

        return m & self._block_layout(blk_shape, head_idx, query_idx, key_idx, blk_idx)


@attr.s
class DenseAttentionMask(AttentionMask):
    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()

        self.global_layout = np.ones([self.n_query_block, self.n_key_block], dtype=np.bool)
        n_zero_query_blocks = self.n_query_pad // self.block_size
        n_zero_key_blocks = self.n_key_pad // self.block_size
        self.global_layout[self.n_query_block - n_zero_query_blocks :] = False
        self.global_layout[:, self.n_key_block - n_zero_key_blocks :] = False

    def _block_layout(
        self, blk_shape: Any, head_idx: int, query_idx: int, key_idx: int, blk_idx: int
    ) -> np.ndarray:
        return np.ones([self.block_size, self.block_size], dtype=np.bool)


@attr.s
class DenseCausalAttentionMask(AttentionMask):
    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()

        self.global_layout = np.tril(np.ones([self.n_query_block, self.n_key_block], dtype=np.bool))
        n_zero_query_blocks = self.n_query_pad // self.block_size
        n_zero_key_blocks = self.n_key_pad // self.block_size
        self.global_layout[self.n_query_block - n_zero_query_blocks :] = False
        self.global_layout[:, self.n_key_block - n_zero_key_blocks :] = False

    def _block_layout(
        self, blk_shape: Any, head_idx: int, query_idx: int, key_idx: int, blk_idx: int
    ) -> np.ndarray:
        if query_idx > key_idx:
            return np.ones(2 * [self.block_size], dtype=np.bool)
        elif query_idx < key_idx:
            return np.zeros(2 * [self.block_size], dtype=np.bool)
        else:
            return np.tril(np.ones(2 * [self.block_size], dtype=np.bool))


@attr.s(eq=False, repr=False)
class AttentionInfo:
    n_heads: int = attr.ib()
    ctx_blks_q: int = attr.ib()
    ctx_blks_k: int = attr.ib()
    block_size: int = attr.ib()
    pytorch_attn_bias: Optional[torch.Tensor] = attr.ib()


def to_attention_info(d: AttentionMask) -> AttentionInfo:
    return AttentionInfo(
        n_heads=d.n_head,
        ctx_blks_q=d.n_query_block,
        ctx_blks_k=d.n_key_block,
        block_size=d.block_size,
        pytorch_attn_bias=None,
    )


def make_full_layout(d: AttentionMask) -> np.ndarray:
    """
    Returns the `context_size x context_size` layout matrix described by `d`. If the layout is dependent on the index of
    the attention head, a `attention_head x context_size x context_size` layout matrix is returned instead.
    """

    if not d.is_head_specific:
        u = np.reshape(d.global_layout, [d.n_query_block, d.n_key_block, 1, 1])
        r = product(range(d.n_query_block), range(d.n_key_block))
        v = np.array([d.block_layout(None, 0, i, j, 0) for i, j in r])
        v = np.reshape(v, [d.n_query_block, d.n_key_block, d.block_size, d.block_size])

        w = u * v
        w = np.transpose(w, [0, 2, 1, 3])
        w = np.reshape(w, [d.query_context_size, d.key_context_size])
        return w
    else:
        if len(d.global_layout.shape) == 2:
            u = np.reshape(d.global_layout, [1, d.n_query_block, d.n_key_block, 1, 1])
            u = np.tile(u, [d.n_head, 1, 1, 1, 1])
        elif len(d.global_layout.shape) == 3:
            u = np.reshape(d.global_layout, [d.n_head, d.n_query_block, d.n_key_block, 1, 1])
        else:
            raise RuntimeError()

        s = product(range(d.n_head), range(d.n_query_block), range(d.n_key_block))
        v = np.array([d.block_layout(None, i, j, k, 0) for i, j, k in s])
        v = np.reshape(v, [d.n_head, d.n_query_block, d.n_key_block, d.block_size, d.block_size])

        w = u * v
        w = np.transpose(w, [0, 1, 3, 2, 4])
        w = np.reshape(w, [d.n_head, d.query_context_size, d.key_context_size])
        return w
