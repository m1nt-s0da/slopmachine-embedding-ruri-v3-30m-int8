from typing import Protocol
from dataclasses import dataclass
import numpy as np

__all__ = ["TokenizerOutput", "EmbeddingModel", "EmbeddingModelLoader"]


@dataclass
class TokenizerOutput:
    input_ids: np.ndarray
    attention_mask: np.ndarray


class EmbeddingModel(Protocol):
    async def tokenize(self, text: list[str]) -> TokenizerOutput: ...
    async def encode(
        self, input_ids: np.ndarray, attention_mask: np.ndarray | None = None
    ) -> np.ndarray: ...


class EmbeddingModelLoader(Protocol):
    async def __aenter__(self) -> EmbeddingModel: ...
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None: ...
