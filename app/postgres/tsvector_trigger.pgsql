
-- Drop the existing trigger if it exists
DROP TRIGGER IF EXISTS document_tsvector_update ON langchain_pg_embedding;

-- Create or replace the function
CREATE OR REPLACE FUNCTION update_document_tsvector() RETURNS trigger AS $$
BEGIN
    NEW.document_tsvector := to_tsvector('anish', NEW.document);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create the trigger
CREATE TRIGGER document_tsvector_update
BEFORE INSERT OR UPDATE ON langchain_pg_embedding
FOR EACH ROW EXECUTE FUNCTION update_document_tsvector();