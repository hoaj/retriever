import os
from langchain_postgres.vectorstores import PGVector
from app.util.util import Util
from app.redis.cache import CacheManager


class VectorStoreManager:
    def __init__(
        self,
        collection_name: str = "vectordb",
    ):
        # Load environment variables
        db_user = os.getenv("DB_USER", "admin")
        db_password = os.getenv("DB_PASSWORD", "admin")
        db_host = "postgres" if os.getenv("ENV", "local") == "compose" else "localhost"
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "vectordb")

        self._connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

        self._vector_store = PGVector(
            embeddings=CacheManager().cached_embeddings,
            connection=self._connection_string,
            collection_name=collection_name,
        )
        self._connection_string_keyword = (
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
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
