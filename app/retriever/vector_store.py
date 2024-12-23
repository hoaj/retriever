from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings


class VectorStoreManager:
    def __init__(
        self,
        connection_string: str = "postgresql+psycopg2://admin:admin@localhost:5433/vectordb",
        collection_name: str = "vectordb",
        embeddings=OpenAIEmbeddings(),
    ):
        self._vector_store = PGVector(
            embeddings=embeddings,
            connection=connection_string,
            collection_name=collection_name,
        )

    @property
    def vector_store(self) -> PGVector:
        """Getter for vector_store."""
        return self._vector_store

    def get_retriever(self, search_kwargs=None):
        """Returns a retriever from the vector store."""
        if search_kwargs is None:
            search_kwargs = {
                "k": 30,
            }  # Default search parameters
        return self._vector_store.as_retriever(search_kwargs=search_kwargs)


# Example usage
if __name__ == "__main__":
    # run with this cmd: python app/retriever/vector_store.py
    from dotenv import load_dotenv
    import os

    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

    vector_store_manager = VectorStoreManager()
    # Access the vector store
    vs = vector_store_manager.vector_store
    print("Vector store initialized:", vs)

    # Get a retriever
    retriever = vector_store_manager.get_retriever()
    print("Retriever initialized:", retriever)
