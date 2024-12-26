import os
import time
from langchain.retrievers import EnsembleRetriever
from app.util.util import Util
from app.postgres.vector_store import VectorStoreManager
from app.retrievers.keyword_retriever import KeywordRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document


class HybridSearch2(BaseRetriever):

    semantic_retriever: VectorStoreRetriever
    keyword_retriever: KeywordRetriever

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.semantic_retriever,
                self.keyword_retriever,
            ],
            weights=[0.5, 0.5],  # Example weights, adjust as needed
            id_key="id",
        )

        results = ensemble_retriever.invoke(query)

        results = Util.get_top_documents(results, 5)
        return results


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv

    # python -m app.retriever.retrieval
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
    vector_store_manager: VectorStoreManager = VectorStoreManager()
    semantic_retriever = vector_store_manager.get_semantic_retriever()
    hybrid = HybridSearch2(
        semantic_retriever=semantic_retriever,
        keyword_retriever=KeywordRetriever(),
    )

    query = "Kan lejer stilles til ansvar for udgifter det ligger udover depositum?"

    start_time = time.time()
    results = hybrid.invoke(query)
    end_time = time.time()
    execution_time = end_time - start_time

    Util.save_data_to_json(results, "hybrid_2_results.json")
    print(f"Execution time for retrieval: {execution_time} seconds")
