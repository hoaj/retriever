SELECT id, document, ts_rank(document_tsvector, websearch_to_tsquery('danish', 'Hvad er lejers rettigheder ift. opsigelse?')) AS rank
FROM langchain_pg_embedding
ORDER BY rank DESC
LIMIT 10;


-- WHERE document_tsvector @@ websearch_to_tsquery('Hvad er lejers rettigheder ift. opsigelse?')