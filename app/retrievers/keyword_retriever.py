from typing import List
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import psycopg2
import psycopg2.extras
import asyncpg
import json


from app.util import Util


class KeywordRetriever(BaseRetriever):
    """A retriever that uses keyword search to find relevant documents."""

    connection_string: str = "postgresql://admin:admin@postgres:5432/vectordb"
    k: int = 10

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementation for retriever using keyword search."""
        conn = psycopg2.connect(self.connection_string)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, cmetadata, document, ts_rank(document_tsvector, websearch_to_tsquery('danish', %s)) AS rank
                    FROM langchain_pg_embedding
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    (query, self.k),
                )
                rows = cur.fetchall()
                documents = []
                for row in rows:
                    documents.append(
                        Document(
                            id=row["id"],
                            page_content=row["document"],
                            metadata=row["cmetadata"],
                            type="Document",
                        )
                    )
                return documents
        finally:
            conn.close()

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query."""
        conn = await asyncpg.connect(self.connection_string)
        try:
            rows = await conn.fetch(
                """
                SELECT id, cmetadata, document, ts_rank(document_tsvector, websearch_to_tsquery('danish', $1)) AS rank
                FROM langchain_pg_embedding
                ORDER BY rank DESC
                LIMIT $2
                """,
                query,
                self.k,
            )
            documents = []
            for row in rows:
                documents.append(
                    Document(
                        id=row["id"],
                        page_content=row["document"],
                        metadata=json.loads(row["cmetadata"]),
                        type="Document",
                    )
                )
            return documents
        finally:
            await conn.close()


if __name__ == "__main__":
    # Test the KeywordRetriever
    import asyncio

    # Initialize the retriever
    retriever = KeywordRetriever()

    # Define a sample query
    sample_query = "Hvor lang tid opsigelse har jeg?"

    # Test the synchronous method
    try:
        documents = retriever.invoke(sample_query)
        Util.save_data_to_json(documents, "synchronous_result_keyword.json")
    except Exception as e:
        print(f"Error during synchronous retrieval: {e}")

    # Test the asynchronous method
    async def test_async_retrieval():
        try:
            documents = await retriever.ainvoke(sample_query)
            Util.save_data_to_json(documents, "asynchronous_result_keyword.json")
        except Exception as e:
            print(f"Error during asynchronous retrieval: {e}")

    # Run the async test
    asyncio.run(test_async_retrieval())
