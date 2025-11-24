import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from .client import EmbedderClient

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
else:
    import contextlib

    with contextlib.suppress(ImportError):
        from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceTransformersEmbedder(EmbedderClient):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        try:
            self.model = SentenceTransformer(model_name)
        except NameError:
            raise ImportError(
                'sentence-transformers is required for SentenceTransformersEmbedder. '
                'Install it with: pip install graphiti-core[sentence-transformers]'
            ) from None

        logger.info(f'Loading SentenceTransformer model: {model_name}')

    def _to_list_float(self, embedding: Any) -> list[float]:
        """Helper to convert any embedding format to list of floats."""
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        if hasattr(embedding, 'tolist'):  # Tensor
            return embedding.tolist()
        if isinstance(embedding, list):
            return [float(x) for x in embedding]
        return list(embedding)

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        # Normalize input
        if isinstance(input_data, (str, int)):
            inp = input_data
        elif (
            isinstance(input_data, list) and len(input_data) == 1 and isinstance(input_data[0], str)
        ):
            inp = input_data[0]
        else:
            inp = input_data

        # Encode
        embedding = self.model.encode(inp, convert_to_numpy=True, convert_to_tensor=False)

        if (
            isinstance(embedding, list)
            and len(embedding) > 0
            and isinstance(embedding[0], (list, np.ndarray))
        ):
            # Matrix
            return self._to_list_float(embedding[0])
        
        # Check if it's a numpy array or tensor with shape
        if hasattr(embedding, 'shape') and len(embedding.shape) > 1:  # 2D array
            return self._to_list_float(embedding[0])

        return self._to_list_float(embedding)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        if not input_data_list:
            return []
        embeddings = self.model.encode(
            input_data_list, convert_to_numpy=True, convert_to_tensor=False
        )
        return [self._to_list_float(emb) for emb in embeddings]