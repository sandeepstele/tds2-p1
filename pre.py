# pre.py - Preprocessing script for knowledge base
import os
import json
import sqlite3
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import aiohttp
import asyncio
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv

# FAISS imports
import faiss
import pickle
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths and constants
DISCOURSE_DIR = "downloaded_threads"
MARKDOWN_DIR = "markdown_files"
DB_PATH = "knowledge_base.db"
FAISS_INDEX_PATH = "knowledge_base.index"
ID_MAPPING_PATH = "id_to_chunk_meta.pkl"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ensure directories exist
os.makedirs(DISCOURSE_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)

# Get API key
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.error("API_KEY environment variable not set. Please set it before running.")

# Database connection
def create_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        logger.info(f"Connected to SQLite database at {DB_PATH}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
        return None

# Create tables
def create_tables(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS discourse_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER,
            topic_id INTEGER,
            topic_title TEXT,
            post_number INTEGER,
            author TEXT,
            created_at TEXT,
            likes INTEGER,
            chunk_index INTEGER,
            content TEXT,
            url TEXT,
            embedding BLOB
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS markdown_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_title TEXT,
            original_url TEXT,
            downloaded_at TEXT,
            chunk_index INTEGER,
            content TEXT,
            embedding BLOB
        )
    ''')
    conn.commit()
    logger.info("Database tables created successfully")

# Chunking logic
def create_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    if not text:
        return []
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) <= chunk_size:
        return [text]
    paragraphs = text.split('\n')
    chunks, current = [], ''
    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(current.strip()); current = ''
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_chunk = ''
            for s in sentences:
                if len(s) > chunk_size:
                    if sent_chunk:
                        chunks.append(sent_chunk.strip()); sent_chunk = ''
                    for i in range(0, len(s), chunk_size - chunk_overlap):
                        part = s[i:i+chunk_size]
                        if part: chunks.append(part.strip())
                elif sent_chunk and len(sent_chunk) + len(s) > chunk_size:
                    chunks.append(sent_chunk.strip()); sent_chunk = s
                else:
                    sent_chunk = sent_chunk + ' ' + s if sent_chunk else s
            if sent_chunk:
                chunks.append(sent_chunk.strip())
        elif current and len(current) + len(para) > chunk_size:
            chunks.append(current.strip()); current = para
        else:
            current = current + ' ' + para if current else para
    if current.strip():
        chunks.append(current.strip())
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev, curr = chunks[i-1], chunks[i]
        if len(prev) > chunk_overlap:
            start = max(0, len(prev) - chunk_overlap)
            br = prev.rfind('. ', start)
            if br != -1 and br > start:
                ov = prev[br+2:]
                if ov and not curr.startswith(ov):
                    curr = ov + ' ' + curr
        overlapped.append(curr)
    return overlapped

# HTML cleanup
def clean_html(html_content):
    soup = BeautifulSoup(html_content or '', 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()
    text = soup.get_text(separator=' ')
    return re.sub(r'\s+', ' ', text).strip()

# Process Discourse JSON
def process_discourse_files(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
    if cursor.fetchone()[0] > 0:
        logger.info("Discourse chunks exist, skipping")
        return
    files = [f for f in os.listdir(DISCOURSE_DIR) if f.endswith('.json')]
    for fn in tqdm(files, desc='Discourse'):
        data = json.load(open(os.path.join(DISCOURSE_DIR, fn), encoding='utf-8'))
        tid, title, slug = data.get('id'), data.get('title',''), data.get('slug','')
        posts = data.get('post_stream',{}).get('posts',[])
        for post in posts:
            content = clean_html(post.get('cooked',''))
            if len(content) < 20: continue
            url = f"https://discourse.onlinedegree.iitm.ac.in/t/{slug}/{tid}/{post.get('post_number')}"
            for idx, chunk in enumerate(create_chunks(content)):
                cursor.execute('''INSERT INTO discourse_chunks
                    (post_id,topic_id,topic_title,post_number,author,created_at,likes,chunk_index,content,url,embedding)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)''',(
                    post.get('id'), tid, title,
                    post.get('post_number'), post.get('username'),
                    post.get('created_at'), post.get('like_count'),
                    idx, chunk, url, None
                ))
    conn.commit()
    logger.info("Processed Discourse files")

# Process Markdown
def process_markdown_files(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
    if cursor.fetchone()[0] > 0:
        logger.info("Markdown chunks exist, skipping")
        return
    files = [f for f in os.listdir(MARKDOWN_DIR) if f.endswith('.md')]
    for fn in tqdm(files, desc='Markdown'):
        text = open(os.path.join(MARKDOWN_DIR, fn), encoding='utf-8').read()
        fm = re.match(r'^---\n(.*?)\n---\n', text, re.DOTALL)
        title, orig, dt = '', '', ''
        if fm:
            fmtext = fm.group(1)
            m = re.search(r'title: "(.*?)"', fmtext)
            title = m.group(1) if m else ''
            m = re.search(r'original_url: "(.*?)"', fmtext)
            orig = m.group(1) if m else ''
            m = re.search(r'downloaded_at: "(.*?)"', fmtext)
            dt = m.group(1) if m else ''
            text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.DOTALL)
        for idx, chunk in enumerate(create_chunks(text)):
            cursor.execute('''INSERT INTO markdown_chunks
                (doc_title,original_url,downloaded_at,chunk_index,content,embedding)
                VALUES (?,?,?,?,?,NULL)''',
                (title, orig, dt, idx, chunk)
            )
    conn.commit()
    logger.info("Processed Markdown files")

# Embeddings + FAISS
async def create_embeddings(api_key):
    if not api_key:
        logger.error("API_KEY missing, cannot embed.")
        return
    conn = create_connection(); cur = conn.cursor()
    cur.execute("SELECT id,content FROM discourse_chunks WHERE embedding IS NULL")
    dch = cur.fetchall()
    cur.execute("SELECT id,content FROM markdown_chunks WHERE embedding IS NULL")
    mch = cur.fetchall()
    async def embed(session, txt, rid, tbl):
        for i in range(3):
            try:
                r = await session.post(
                    "https://aipipe.org/openai/v1/embeddings",
                    headers={"Authorization":api_key,"Content-Type":"application/json"},
                    json={"model":"text-embedding-3-small","input":txt}
                )
                if r.status == 200:
                    d = await r.json()
                    vec = d['data'][0]['embedding']
                    blob = json.dumps(vec).encode()
                    cur.execute(f"UPDATE {tbl} SET embedding = ? WHERE id = ?", (blob, rid))
                    conn.commit()
                    return vec
                elif r.status == 429:
                    await asyncio.sleep(2 ** i)
                else:
                    break
            except Exception as e:
                logger.error(e); await asyncio.sleep(1)
        return None
    async with aiohttp.ClientSession() as sess:
        for batch, tbl in ((dch,'discourse_chunks'), (mch,'markdown_chunks')):
            for i in range(0, len(batch), 10):
                tasks = [embed(sess, row['content'], row['id'], tbl) for row in batch[i:i+10]]
                await asyncio.gather(*tasks)
                await asyncio.sleep(1)
    conn.close()
    
    # Build FAISS index
    index = faiss.IndexFlatIP(1536)
    id_map = {}
    conn2 = create_connection(); c2 = conn2.cursor()
    # Add from discourse_chunks
    c2.execute("SELECT id, embedding, url, chunk_index FROM discourse_chunks WHERE embedding IS NOT NULL")
    for fid, row in enumerate(c2.fetchall()):
        rid, raw, url, ci = row['id'], row['embedding'], row['url'], row['chunk_index']
        vec = np.array(json.loads(raw.decode()), dtype='float32')
        faiss.normalize_L2(vec.reshape(1, -1))
        index.add(vec.reshape(1, -1))
        id_map[fid] = {'table': 'discourse', 'row_id': rid, 'url': url, 'chunk_index': ci}
    # Add from markdown_chunks
    c2.execute("SELECT id, embedding, original_url, chunk_index FROM markdown_chunks WHERE embedding IS NOT NULL")
    for fid2, row in enumerate(c2.fetchall(), start=len(id_map)):
        rid, raw, orig_url, ci = row['id'], row['embedding'], row['original_url'], row['chunk_index']
        vec = np.array(json.loads(raw.decode()), dtype='float32')
        faiss.normalize_L2(vec.reshape(1, -1))
        index.add(vec.reshape(1, -1))
        id_map[fid2] = {'table': 'markdown', 'row_id': rid, 'url': orig_url, 'chunk_index': ci}
    conn2.close()
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(ID_MAPPING_PATH, 'wb') as f:
        pickle.dump(id_map, f)
    logger.info("Saved FAISS index and mapping")

async def main():
    global CHUNK_SIZE, CHUNK_OVERLAP
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()
    CHUNK_SIZE, CHUNK_OVERLAP = args.chunk_size, args.chunk_overlap
    
    conn = create_connection()
    create_tables(conn)
    process_discourse_files(conn)
    process_markdown_files(conn)
    conn.close()
    await create_embeddings(API_KEY)
    logger.info("Preprocessing & FAISS index build complete.")

if __name__ == '__main__':
    asyncio.run(main())
