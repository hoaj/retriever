import os
import time
from langchain.retrievers import EnsembleRetriever
from app.helpers.util import GobalUtil
from app.postgres.vector_store import VectorStoreManager
from app.retrievers.keyword_retriever import KeywordRetriever


class HybridSearch:
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()

        self.semantic_retriever = self.vector_store_manager.get_semantic_retriever()
        self.keyword_retriever = KeywordRetriever()

    def retrieve(self, query: str):
        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.semantic_retriever,
                self.keyword_retriever,
            ],
            weights=[0.5, 0.5],  # Example weights, adjust as needed
            id_key="id",
        )

        results = ensemble_retriever.invoke(query)

        results = GobalUtil.get_top_documents(results, 5)
        return results


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv

    # python -m app.retriever.retrieval
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

    hybrid = HybridSearch()

    query = "Kan lejer stilles til ansvar for udgifter det ligger udover depositum?"

    start_time = time.time()
    results = hybrid.retrieve(query)
    end_time = time.time()
    execution_time = end_time - start_time

    GobalUtil.save_data_to_json(results, "hybrid_2_results.json")
    print(f"Execution time for retrieval: {execution_time} seconds")
