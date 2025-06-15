import os
import json
import sqlite3
import numpy as np
import re
import faiss
import pickle
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import aiohttp
import asyncio
import logging
import base64
from fastapi.responses import JSONResponse
import uvicorn
import traceback
from dotenv import load_dotenv

# Load environment and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "knowledge_base.db"
FAISS_INDEX_PATH = "knowledge_base.index"
ID_MAPPING_PATH = "id_to_chunk_meta.pkl"
SIMILARITY_THRESHOLD = 0.22
MAX_RESULTS = 10
MAX_CONTEXT_CHUNKS = 4
API_KEY = os.getenv("API_KEY")

# Load FAISS index and ID→meta mapping
logger.info("Loading FAISS index and metadata mapping...")
_index = faiss.read_index(FAISS_INDEX_PATH)
with open(ID_MAPPING_PATH, "rb") as f:
    _id_map = pickle.load(f)
logger.info("FAISS index and mapping loaded")

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 jpeg

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# Initialize FastAPI
app = FastAPI(title="RAG Query API (FAISS+RAG)", description="FAISS retrieval + RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# Database helper
def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"DB connect error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

# Embedding
async def get_embedding(text: str, max_retries: int = 3) -> List[float]:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not set")
    retries = 0
    while True:
        try:
            url = "https://aipipe.org/openai/v1/embeddings"
            headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
            payload = {"model": "text-embedding-3-small", "input": text}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["data"][0]["embedding"]
                    elif resp.status == 429 and retries < max_retries:
                        await asyncio.sleep(2 ** retries)
                        retries += 1
                        continue
                    else:
                        body = await resp.text()
                        raise HTTPException(status_code=resp.status, detail=body)
        except Exception as e:
            if retries < max_retries:
                retries += 1
                await asyncio.sleep(1)
                continue
            logger.error(f"Embedding error: {e}")
            raise HTTPException(status_code=500, detail="Embedding failed")

# Multimodal support
async def process_multimodal_query(question: str, image_base64: Optional[str]) -> List[float]:
    if not image_base64:
        return await get_embedding(question)
    # Send to Vision LLM
    logger.info("Processing multimodal input")
    url = "https://aipipe.org/openai/v1/chat/completions"
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    image_uri = f"data:image/jpeg;base64,{image_base64}"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Describe what you see relevant to: {question}"},
                    {"type": "image_url", "image_url": {"url": image_uri}}
                ]
            }
        ]
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                desc = data["choices"][0]["message"]["content"]
                combined = f"{question}\nImage context: {desc}"
                return await get_embedding(combined)
    # fallback
    return await get_embedding(question)

# FAISS retrieval
def find_similar_chunks(query_vec: List[float]) -> List[dict]:
    q = np.array(query_vec, dtype='float32').reshape(1, -1)
    faiss.normalize_L2(q)
    distances, ids = _index.search(q, MAX_RESULTS)
    hits = []
    conn = get_db_connection(); cur = conn.cursor()
    for score, fid in zip(distances[0], ids[0]):
        if score < SIMILARITY_THRESHOLD: continue
        meta = _id_map.get(int(fid))
        if not meta: continue
        tbl = "discourse_chunks" if meta["table"] == "discourse" else "markdown_chunks"
        urlcol = "url" if meta["table"]=="discourse" else "original_url"
        cur.execute(f"SELECT content, {urlcol}, chunk_index FROM {tbl} WHERE id=?", (meta["row_id"],))
        row = cur.fetchone()
        if not row: continue
        hits.append({
            "source": meta["table"],
            "id": meta["row_id"],
            "chunk_index": row[2],
            "url": row[1],
            "content": row[0],
            "similarity": float(score)
        })
    conn.close()
    # sort & limit
    hits.sort(key=lambda x: x["similarity"], reverse=True)
    return hits[:MAX_CONTEXT_CHUNKS]

# Enrich adjacent chunks
async def enrich_with_adjacent_chunks(results: List[dict]) -> List[dict]:
    conn = get_db_connection(); cur = conn.cursor()
    enriched = []
    for r in results:
        content = r["content"]
        extra = ""
        if r["source"] == "discourse":
            cur.execute("SELECT content FROM discourse_chunks WHERE post_id=(SELECT post_id FROM discourse_chunks WHERE id=?) AND chunk_index=?",
                        (r["id"], r["chunk_index"]-1))
            prev = cur.fetchone()
            if prev: extra += prev[0] + " "
            cur.execute("SELECT content FROM discourse_chunks WHERE post_id=(SELECT post_id FROM discourse_chunks WHERE id=?) AND chunk_index=?",
                        (r["id"], r["chunk_index"]+1))
            nxt = cur.fetchone()
            if nxt: extra += nxt[0]
        else:
            cur.execute("SELECT content FROM markdown_chunks WHERE doc_title=(SELECT doc_title FROM markdown_chunks WHERE id=?) AND chunk_index=?",
                        (r["id"], r["chunk_index"]-1))
            prev = cur.fetchone()
            if prev: extra += prev[0] + " "
            cur.execute("SELECT content FROM markdown_chunks WHERE doc_title=(SELECT doc_title FROM markdown_chunks WHERE id=?) AND chunk_index=?",
                        (r["id"], r["chunk_index"]+1))
            nxt = cur.fetchone()
            if nxt: extra += nxt[0]
        if extra: r["content"] = content + " " + extra
        enriched.append(r)
    conn.close()
    return enriched

# Prompt builder & answer generation
async def generate_answer(question: str, contexts: List[dict]) -> str:
    ctx = "".join([f"\n\n{c['source'].capitalize()} (URL: {c['url']}):\n{c['content'][:1500]}" for c in contexts])
    prompt = (
        "Answer the question using ONLY the context below. "
        "If you cannot, say you don't have enough information.\n\n"
        f"Context:{ctx}\n\nQuestion: {question}\n\n"
        "Format:\n1. Answer\n2. Sources:\n1. URL: [url], Text: [snippet]\n"
    )
    payload = {"model":"gpt-4o-mini","messages":[
        {"role":"system","content":"You are a helpful assistant. Use only provided context."},
        {"role":"user","content":prompt}
    ],"temperature":0.3}
    url = "https://aipipe.org/openai/v1/chat/completions"
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status==200:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
            text = await resp.text()
            raise HTTPException(status_code=resp.status, detail=text)

# Parse LLM response
def parse_llm_response(response: str) -> Dict[str,Any]:
    parts = response.split("Sources:",1)
    answer = parts[0].strip()
    links = []
    if len(parts)>1:
        for line in parts[1].splitlines():
            m_url = re.search(r"URL:\s*\[(.*?)\]",line)
            m_text= re.search(r"Text:\s*\[(.*?)\]",line)
            if m_url:
                links.append({"url":m_url.group(1),"text":m_text.group(1) if m_text else ""})
    return {"answer":answer,"links":links}

# API endpoints
@app.post("/query")
async def query(request: QueryRequest) -> QueryResponse:
    try:
        emb = await process_multimodal_query(request.question, request.image)
        hits = find_similar_chunks(emb)
        if not hits:
            return {"answer":"No relevant info found.","links":[]}
        enriched = await enrich_with_adjacent_chunks(hits)
        resp = await generate_answer(request.question, enriched)
        parsed = parse_llm_response(resp)
        # fallback links
        if not parsed["links"]:
            parsed["links"]=[{"url":h["url"],"text":h["content"][:100]+"..."} for h in enriched[:5]]
        return parsed
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal error")

@app.get("/health")
def health():
    try:
        conn = sqlite3.connect(DB_PATH); cur=conn.cursor()
        cur.execute("SELECT COUNT(*) FROM discourse_chunks"); dcnt=cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM markdown_chunks"); mcnt=cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL"); dec=cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL"); mec=cur.fetchone()[0]
        conn.close()
        return {"status":"healthy","db":"connected","faiss_loaded":bool(_index),
                "discourse_chunks":dcnt,"markdown_chunks":mcnt,
                "discourse_embeddings":dec,"markdown_embeddings":mec,
                "api_key_set":bool(API_KEY)}
    except Exception as e:
        logger.error(f"Health error: {e}")
        return JSONResponse(status_code=500, content={"status":"unhealthy","error":str(e)})

if __name__ == "__main__":
    uvicorn.run("app:app",host="0.0.0.0",port=8000,reload=True)
