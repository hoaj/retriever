from fastapi import APIRouter
from langserve import add_routes
from app.retrievers.hybrid_1 import HybridSearch1
from app.retrievers.hybrid_2 import HybridSearch2
from app.retrievers.keyword_retriever import KeywordRetriever
from app.postgres.vector_store import VectorStoreManager


def setup_routes() -> APIRouter:
    router = APIRouter()

    vector_store_manager: VectorStoreManager = VectorStoreManager()
    semantic_retriever = vector_store_manager.get_semantic_retriever()
    keyword_retriever = KeywordRetriever()

    hybrid1 = HybridSearch1(
        semantic_retriever=semantic_retriever,
        keyword_retriever=keyword_retriever,
    )

    hybrid2 = HybridSearch2(
        semantic_retriever=semantic_retriever,
        keyword_retriever=keyword_retriever,
    )

    add_routes(router, hybrid1, path="/hybrid1")
    add_routes(router, hybrid2, path="/hybrid2")

    return router
