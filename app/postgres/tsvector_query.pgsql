SELECT id, document, ts_rank(document_tsvector, plainto_tsquery('danish', 'Hvad er lejers rettigheder ift. opsigelse?')) AS rank
FROM langchain_pg_embedding
-- WHERE document_tsvector @@ plainto_tsquery('Hvad er lejers rettigheder ift. opsigelse?')
ORDER BY rank DESC
LIMIT 10;

