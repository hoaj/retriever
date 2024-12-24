from typing import List
from langchain_community.storage import RedisStore
from langchain_community.utilities.redis import get_client
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings


class CacheManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        underlying_embeddings=OpenAIEmbeddings(),
    ):
        if self._initialized:
            return
        self._client = get_client(redis_url)
        self._store = RedisStore(client=self._client)
        self._cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings,
            self._store,
            namespace="cached_embeddings:",
            query_embedding_cache=True,
        )
        self._initialized = True

    def set_values(self, key_value_pairs):
        """Set multiple key-value pairs in the Redis store."""
        self._store.mset(key_value_pairs)

    def get_values(self, keys):
        """Get values for multiple keys from the Redis store."""
        return self._store.mget(keys)

    def delete_keys(self, keys):
        """Delete multiple keys from the Redis store."""
        self._store.mdelete(keys)

    def iterate_keys(self):
        """Iterate over all keys in the Redis store."""
        for key in self._store.yield_keys():
            print(key)  # noqa: T201

    def embed_query(self, query: str) -> List[float]:
        """Wrapper for embedding a query using cached embeddings."""
        return self._cached_embeddings.embed_query(query)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Wrapper for embedding documents using cached embeddings."""
        return self._cached_embeddings.embed_documents(documents)

    @property
    def cached_embeddings(self) -> CacheBackedEmbeddings:
        """Getter for cached_embeddings."""
        return self._cached_embeddings


if __name__ == "__main__":
    # run with this cmd: python app/retriever/cache.py
    from dotenv import load_dotenv
    import os

    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

    # Test code to verify embedding a query
    cache_manager = CacheManager()
    test_query = "What is the capital of France?"
    embedded_query = cache_manager.embed_query(test_query)
    print(f"Embedded query for '{test_query}': {embedded_query}")

    # Test set_values and iterate_keys
    key_value_pairs = [("key1", b"value1"), ("key2", b"value2")]
    cache_manager.set_values(key_value_pairs)

    print("Keys in Redis store after setting values:")
    cache_manager.iterate_keys()
