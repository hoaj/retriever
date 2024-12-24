import os
import time
from langchain_cohere import CohereRerank
from langchain_openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from app.helpers.util import GobalUtil
from langchain_core.documents import Document
from langchain.retrievers import (
    ContextualCompressionRetriever,
    MergerRetriever,
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
)
from langchain_community.document_transformers import LongContextReorder
from app.postgres.vector_store import VectorStoreManager
from app.helpers.cache import CacheManager
from app.retrievers.keyword_retriever import KeywordRetriever


class HybridSearch:
    def __init__(
        self,
    ):
        # self.vector_store_manager = VectorStoreManager()

        # self.semantic_retriever = self.vector_store_manager.get_semantic_retriever()
        self.keyword_retriever = KeywordRetriever()

    def retrieve(self, query: str):

        lotr = MergerRetriever(
            retrievers=[
                # self.semantic_retriever,
                self.keyword_retriever,
            ]
        )

        filter = EmbeddingsRedundantFilter(embeddings=CacheManager().cached_embeddings)
        cohere_rerank_model = CohereRerank(
            model="rerank-multilingual-v3.0",
            top_n=10,
        )

        reordering = LongContextReorder()
        pipeline = DocumentCompressorPipeline(
            transformers=[
                filter,
                reordering,
                cohere_rerank_model,
            ]
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline, base_retriever=lotr
        )
        reranked_combined_results = compression_retriever.invoke(query)
        return reranked_combined_results


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    # python -m app.retriever.retrieval
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

    hybrid = HybridSearch()

    query = "Hvad er mine rettigheder ift opsigelse af lejemål"

    start_time = time.time()
    results = hybrid.retrieve(query)
    end_time = time.time()
    execution_time = end_time - start_time

    GobalUtil.save_data_to_json(results, "doc_results.json")
    print(f"Execution time for retrieval: {execution_time} seconds")
