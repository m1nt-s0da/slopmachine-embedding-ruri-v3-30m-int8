from slopmachine.embedding import EmbeddingModel, TokenizerOutput, EmbeddingModelLoader
import os
from tokenizers import Tokenizer, Encoding
from onnxruntime import InferenceSession
import numpy as np

__all__ = ["LaionClapInt8"]


class LaionClapInt8(EmbeddingModel, EmbeddingModelLoader):
    def __init__(
        self,
        /,
        model_path: str | os.PathLike[str] | None = None,
        tokenizer_path: str | os.PathLike[str] | None = None,
    ):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "model.onnx")
        if tokenizer_path is None:
            tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")

        self.__tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.__tokenizer.enable_padding()

        self.__encoder = InferenceSession(str(model_path))

    async def __aenter__(self) -> EmbeddingModel:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    async def tokenize(self, text: list[str]) -> TokenizerOutput:
        encoded: list[Encoding] = await self.__tokenizer.async_encode_batch(text)

        return TokenizerOutput(
            input_ids=np.array([e.ids for e in encoded], dtype=np.int64),
            attention_mask=np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            ),
        )

    async def encode(
        self, input_ids: np.ndarray, attention_mask: np.ndarray | None = None
    ) -> np.ndarray:
        if attention_mask is None:
            attention_mask = np.ones_like(input_ids, dtype=np.int64)

        onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = self.__encoder.run(None, onnx_inputs)
        last_hidden_state = outputs[0]

        sum_embeddings = np.sum(
            last_hidden_state * np.expand_dims(attention_mask, axis=-1), axis=1
        )
        sum_mask = np.sum(attention_mask, axis=1, keepdims=True)
        embeddings = sum_embeddings / sum_mask
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norm

        return embeddings
