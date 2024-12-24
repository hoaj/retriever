ALTER TABLE langchain_pg_embedding ADD COLUMN document_tsvector tsvector;

UPDATE langchain_pg_embedding SET document_tsvector = to_tsvector('danish', document);

CREATE INDEX idx_document_tsvector ON langchain_pg_embedding USING GIN(document_tsvector);
