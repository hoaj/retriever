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
from app.retriever.vector_store import VectorStoreManager
from app.retriever.cache import CacheManager


class RetrievalManager:
    def __init__(
        self,
        splits_filename: str = "splits.json",
    ):
        self.splits_filepath = "app/retriever/data/" + splits_filename

        # Initialize CacheManager
        self.cache_manager = CacheManager()

        # Initialize VectorStoreManager
        self.vector_store_manager = VectorStoreManager(
            embeddings=self.cache_manager.cached_embeddings,
        )

        # Initialize BM25 retriever
        self.bm25_retriever = None

    def add_docs_to_vector_store(self):
        docs = GobalUtil.load_docs(self.splits_filepath)
        self.vector_store_manager.get_retriever().add_documents(documents=docs)

    def add_docs_to_bm25_retriever(self):
        docs = GobalUtil.load_docs(self.splits_filepath)
        self.bm25_retriever = BM25Retriever.from_documents(docs)

    def retrieve(self, query: str):

        lotr = MergerRetriever(
            retrievers=[
                self.vector_store_manager.get_retriever(),
                self.bm25_retriever,
            ]
        )

        filter = EmbeddingsRedundantFilter(
            embeddings=self.cache_manager.cached_embeddings
        )
        cohere_rerank_model = CohereRerank(
            model="rerank-multilingual-v3.0",
            top_n=5,
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

    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

    retrieval_manager = RetrievalManager()

    retrieval_manager.add_docs_to_bm25_retriever()
    # retrieval_manager.add_docs_to_vector_store()

    query = "Hver er lejers ret ift. opsigelse?"

    start_time = time.time()
    results = retrieval_manager.retrieve(query)
    end_time = time.time()
    execution_time = end_time - start_time

    GobalUtil.save_data_to_json(results, "doc_results.json")
    print(f"Execution time for retrieval: {execution_time} seconds")
