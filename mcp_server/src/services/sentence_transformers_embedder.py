from typing import Iterable, List, Union, Any
import logging
import numpy as np

from graphiti_core.embedder.client import EmbedderClient

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

class SentenceTransformersEmbedder(EmbedderClient):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers is not installed.")
        
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def _to_list_float(self, embedding: Any) -> List[float]:
        """Helper to convert any embedding format to list of floats."""
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        if hasattr(embedding, 'tolist'): # Tensor
            return embedding.tolist()
        if isinstance(embedding, list):
            return [float(x) for x in embedding]
        return list(embedding)

    async def create(
        self, input_data: Union[str, List[str], Iterable[int], Iterable[Iterable[int]]]
    ) -> List[float]:
        # Normalize input
        if isinstance(input_data, (str, int)):
            inp = input_data
        elif isinstance(input_data, list) and len(input_data) == 1 and isinstance(input_data[0], str):
             inp = input_data[0]
        else:
             inp = input_data

        # Encode
        embedding = self.model.encode(inp, convert_to_numpy=True, convert_to_tensor=False)
        
        # If input was a list of strings, encode returns a matrix (list of embeddings)
        # We want a single embedding if input was treated as single item? 
        # But EmbedderClient.create expects single list[float] return.
        # If input is list[str], OpenAI embedder treats it as batch?
        # Wait, OpenAIEmbedder.create takes "str | list[str]" but returns "list[float]".
        # This implies it returns ONE embedding.
        # If input is list[str], it usually means "list of tokens" for OpenAI?
        # No, OpenAI create(input="text") -> embedding. create(input=["text"]) -> [embedding].
        
        # But the return type annotation of `create` is `list[float]`.
        # This implies it ALWAYS returns a SINGLE vector.
        # So if input is list[str], maybe it expects it to be joined?
        # Or maybe `input_data` as list[str] is treated as tokens?
        
        # Let's check how it's used. Graphiti likely passes a string.
        
        if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], (list, np.ndarray)):
             # Matrix
             return self._to_list_float(embedding[0])
        if len(embedding.shape) > 1: # 2D array
             return self._to_list_float(embedding[0])
             
        return self._to_list_float(embedding)

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        if not input_data_list:
            return []
        embeddings = self.model.encode(input_data_list, convert_to_numpy=True, convert_to_tensor=False)
        return [self._to_list_float(emb) for emb in embeddings]
