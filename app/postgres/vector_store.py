import psycopg2
import psycopg2.extras
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

from app.util import Util
from app.redis.cache import CacheManager


class VectorStoreManager:
    def __init__(
        self,
        connection_string: str = "postgresql+psycopg2://admin:admin@localhost:5433/vectordb",
        collection_name: str = "vectordb",
    ):
        self._connection_string = connection_string
        self._vector_store = PGVector(
            embeddings=CacheManager().cached_embeddings,
            connection=connection_string,
            collection_name=collection_name,
        )
        self._connection_string_keyword = (
            "postgresql://admin:admin@localhost:5433/vectordb"
        )

    def get_semantic_retriever(self, search_kwargs=None):
        """Returns a retriever from the vector store."""
        if search_kwargs is None:
            search_kwargs = {
                "k": 10,
            }  # Default search parameters
        return self._vector_store.as_retriever(search_kwargs=search_kwargs)


# Example usage
if __name__ == "__main__":
    # run with this cmd: python -m app.retriever.vector_store
    from dotenv import load_dotenv
    from app.redis.cache import CacheManager
    import os

    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

    vector_store_manager = VectorStoreManager()
    # Get a retriever
    retriever = vector_store_manager.get_semantic_retriever()
    print("Retriever initialized:", retriever)
    result_keyword = vector_store_manager.keyword_search("Opsigelse")
    result_semantic = vector_store_manager._vector_store.search(
        "Opsigelse",
        "mmr",
    )
    Util.save_data_to_json(result_keyword, "result_keyword.json")
    Util.save_data_to_json(result_semantic, "result_semantic.json")

    # docs = GobalUtil.load_docs("app/retriever/data/splits.json")
    # retriever.add_documents(
    #     documents=docs,
    #     ids=[doc.metadata["id"] for doc in docs],
    # )
