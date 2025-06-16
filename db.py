#!/usr/bin/env python3
import os
import sqlite3
import json
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

#─── Load env ────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY      = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
parts = PINECONE_ENV.split('-')
if len(parts) == 2:
    region, cloud = parts
else:
    region = PINECONE_ENV
    cloud = "aws"
print(f"Parsed Pinecone spec: cloud='{cloud}', region='{region}'")
DB           = "knowledge_base.db"
INDEX        = "rag-index"
DIM          = 1536
BATCH        = 100

if not API_KEY or not PINECONE_ENV:
    raise RuntimeError("Set PINECONE_API_KEY and PINECONE_ENV in your .env")

print(f"Using PINECONE_ENV='{PINECONE_ENV}'")
print("Existing indexes before deletion:", Pinecone(api_key=API_KEY, spec=ServerlessSpec(cloud=cloud, region=region)).list_indexes().names())

#─── Initialize Pinecone client ─────────────────────────────────────────────
pc = Pinecone(
    api_key=API_KEY,
    spec=ServerlessSpec(cloud=cloud, region=region)
)

try:
    pc.delete_index(name=INDEX)
    print(f"Deleted existing index '{INDEX}'")
except Exception as e:
    print(f"No existing index to delete or delete failed: {e}")

print(f"Creating index '{INDEX}' at dimension={DIM}…")
pc.create_index(
    name=INDEX,
    dimension=DIM,
    metric="cosine",
    spec=ServerlessSpec(cloud=cloud, region=region)
)

index_client = pc.Index(INDEX)

#─── Helper functions ───────────────────────────────────────────────────────
def fetch_rows(table):
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table} WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    conn.close()
    return rows

def chunked(xs, size):
    for i in range(0, len(xs), size):
        yield xs[i : i + size]

def migrate(table, namespace):
    rows = fetch_rows(table)
    print(f"[{table}] → {len(rows)} rows → namespace='{namespace}'")
    for batch in chunked(rows, BATCH):
        vectors = []
        for row in batch:
            emb = json.loads(row["embedding"])
            meta = (
                {
                    "source":    "discourse",
                    "post_id":   row["post_id"],
                    "topic_id":  row["topic_id"],
                    "title":     row["topic_title"],
                    "url":       row["url"],
                    "chunk_idx": row["chunk_index"],
                }
                if table == "discourse_chunks"
                else {
                    "source":       "markdown",
                    "doc_title":    row["doc_title"],
                    "original_url": row["original_url"],
                    "chunk_idx":    row["chunk_index"],
                }
            )
            vectors.append((str(row["id"]), emb, meta))

        # ←— the only change:
        index_client.upsert(vectors=vectors, namespace=namespace)
        print(f" • upserted {len(vectors)} vectors")
        time.sleep(0.1)

#─── Run migration ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    migrate("discourse_chunks", namespace="discourse")
    migrate("markdown_chunks", namespace="markdown")
    print("✅ Migration complete.")