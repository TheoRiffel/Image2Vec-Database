CREATE EXTENSION vector;

CREATE TABLE IF NOT EXISTS metadata (
    id character(24) PRIMARY KEY,
    country_name character varying(30),
    country_id character(2),
    region_id character(2),
    img_rel_path character(55),
    topics character varying(50)[],
    place character varying(20),
    income numeric(10, 3),
    imagenet_synonyms character varying(50)[],
    imagenet_sysnet_id integer[]
);

CREATE TABLE IF NOT EXISTS img_pgvector (
    id character(24) PRIMARY KEY,
    embedding vector(4096),
    CONSTRAINT metadata_id_fkey
        FOREIGN KEY (id)
        REFERENCES metadata (id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS img_pgvector_clip (
    id character(24) PRIMARY KEY,
    embedding vector(768),
    CONSTRAINT metadata_id_fkey
        FOREIGN KEY (id)
        REFERENCES metadata (id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS img_pgarray (
    id character(24) PRIMARY KEY,
    embedding float8[],
    CONSTRAINT metadata_id_fkey
        FOREIGN KEY (id)
        REFERENCES metadata (id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS img_pgarray_clip (
    id character(24) PRIMARY KEY,
    embedding float8[],
    CONSTRAINT metadata_id_fkey
        FOREIGN KEY (id)
        REFERENCES metadata (id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS country_name_idx ON metadata USING HASH (country_name);
CREATE INDEX IF NOT EXISTS region_id_idx ON metadata USING HASH (region_id);
CREATE INDEX IF NOT EXISTS income_idx ON metadata (income);

CREATE INDEX hnsw_idx ON img_pgvector_clip USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ivfflat_idx ON img_pgvector_clip USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
