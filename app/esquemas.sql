CREATE EXTENSION vector;

CREATE TABLE IF NOT EXISTS metadata (
    id character(24) PRIMARY KEY,
    country_name character varying(30),
    country_id character(2),
    region_id character(2),
    img_rel_path character(55),
    topics character varying(50)[],
    place character varying(20),
    income numeric(3,0),
    imagenet_synonyms character varying(50)[],
    imagenet_sysnet_id integer[]
);

CREATE TABLE IF NOT EXISTS img_pgvector (
    id character(24) PRIMARY KEY,
    embedding vector(4096)
);