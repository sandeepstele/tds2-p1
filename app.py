
# app.py
import os
import json
import sqlite3
import numpy as np
import re
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
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Constants
DB_PATH = "knowledge_base.db"
SIMILARITY_THRESHOLD = 0.45  # Lowered threshold for better recall
MAX_RESULTS = 10  # Increased to get more context
load_dotenv()

# Pinecone vector store setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV      = os.getenv("PINECONE_ENV", "us-east-1")
parts = PINECONE_ENV.split('-')
if len(parts) == 2:
    region, cloud = parts
else:
    region = PINECONE_ENV
    cloud = "aws"
pc = Pinecone(api_key=PINECONE_API_KEY,
              spec=ServerlessSpec(cloud=cloud, region=region))
index_client = pc.Index("rag-index")
MAX_CONTEXT_CHUNKS = 4  # Increased number of chunks per source
API_KEY = os.getenv("API_KEY")  # Get API key from environment variable
logger.debug(f"Loaded API_KEY set: {bool(API_KEY)}")
logger.debug(f"PINECONE_ENV: {PINECONE_ENV}, region: {region}, cloud: {cloud}")

# Models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# Initialize FastAPI app
app = FastAPI(title="RAG Query API", description="API for querying the RAG knowledge base")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verify API key is set
if not API_KEY:
    logger.error("API_KEY environment variable is not set. The application will not function correctly.")

# Create a connection to the SQLite database
def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    except sqlite3.Error as e:
        error_msg = f"Database connection error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# Make sure database exists or create it
if not os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Create discourse_chunks table
    c.execute('''
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
    
    # Create markdown_chunks table
    c.execute('''
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
    conn.close()

# Vector similarity calculation with improved handling
def cosine_similarity(vec1, vec2):
    try:
        # Convert to numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Handle zero vectors
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
            
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
            
        return dot_product / (norm_vec1 * norm_vec2)
    except Exception as e:
        logger.error(f"Error in cosine_similarity: {e}")
        logger.error(traceback.format_exc())
        return 0.0  # Return 0 similarity on error rather than crashing

# Function to get embedding from aipipe proxy with retry mechanism
async def get_embedding(text, max_retries=3):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Getting embedding for text (length: {len(text)})")
            # Call the embedding API through aipipe proxy
            url = "https://aipipe.org/openai/v1/embeddings"
            headers = {
                "Authorization": API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "model": "text-embedding-3-small",
                "input": text
            }
            
            logger.info("Sending request to embedding API")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received embedding")
                        return result["data"][0]["embedding"]
                    elif response.status == 429:  # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(5 * (retries + 1))  # Exponential backoff
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error getting embedding (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except Exception as e:
            error_msg = f"Exception getting embedding (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(3 * retries)  # Wait before retry

# Pinecone vector search for similar content
async def find_similar_content(query_embedding, conn):
    """Fetch top-K nearest vectors from Pinecone across both namespaces. Also fetches content from SQLite."""
    logger.debug(f"Querying Pinecone with embedding of length {len(query_embedding)}")
    results = []
    try:
        for namespace in ["discourse", "markdown"]:
            resp = index_client.query(
                vector=query_embedding,
                top_k=MAX_RESULTS,
                namespace=namespace,
                include_metadata=True
            )
            for match in resp.matches:
                sim = match.score
                if sim < SIMILARITY_THRESHOLD:
                    continue
                md = match.metadata
                # Fetch the original content from SQLite
                table = "discourse_chunks" if namespace == "discourse" else "markdown_chunks"
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT content FROM {table} WHERE id = ?",
                    (int(match.id),)
                )
                row = cursor.fetchone()
                content = row["content"] if row else ""
                results.append({
                    "source":       md.get("source"),
                    "id":           int(match.id),
                    "title":        md.get("title") or md.get("doc_title"),
                    "url":          md.get("url") or md.get("original_url"),
                    "chunk_index":  md.get("chunk_idx"),
                    "similarity":   sim,
                    "content":      content
                })
        logger.debug(f"Found {len(results)} raw Pinecone matches")
        # Sort and limit
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:MAX_RESULTS]
    except Exception as e:
        logger.debug(f"Exception in find_similar_content: {e}")
        raise

# Function to enrich content with adjacent chunks
# async def enrich_with_adjacent_chunks(conn, results):
#     try:
#         logger.info(f"Enriching {len(results)} results with adjacent chunks")
#         cursor = conn.cursor()
#         enriched_results = []
#         
#         for result in results:
#             enriched_result = result.copy()
#             additional_content = ""
#             
#             # Try to get adjacent chunks for context
#             if result["source"] == "discourse":
#                 post_id = result["post_id"]
#                 current_chunk_index = result["chunk_index"]
#                 
#                 # Try to get previous chunk
#                 if current_chunk_index > 0:
#                     cursor.execute("""
#                     SELECT content FROM discourse_chunks 
#                     WHERE post_id = ? AND chunk_index = ?
#                     """, (post_id, current_chunk_index - 1))
#                     prev_chunk = cursor.fetchone()
#                     if prev_chunk:
#                         additional_content = prev_chunk["content"] + " "
#                 
#                 # Try to get next chunk
#                 cursor.execute("""
#                 SELECT content FROM discourse_chunks 
#                 WHERE post_id = ? AND chunk_index = ?
#                 """, (post_id, current_chunk_index + 1))
#                 next_chunk = cursor.fetchone()
#                 if next_chunk:
#                     additional_content += " " + next_chunk["content"]
#                 
#             elif result["source"] == "markdown":
#                 title = result["title"]
#                 current_chunk_index = result["chunk_index"]
#                 
#                 # Try to get previous chunk
#                 if current_chunk_index > 0:
#                     cursor.execute("""
#                     SELECT content FROM markdown_chunks 
#                     WHERE doc_title = ? AND chunk_index = ?
#                     """, (title, current_chunk_index - 1))
#                     prev_chunk = cursor.fetchone()
#                     if prev_chunk:
#                         additional_content = prev_chunk["content"] + " "
#                 
#                 # Try to get next chunk
#                 cursor.execute("""
#                 SELECT content FROM markdown_chunks 
#                 WHERE doc_title = ? AND chunk_index = ?
#                 """, (title, current_chunk_index + 1))
#                 next_chunk = cursor.fetchone()
#                 if next_chunk:
#                     additional_content += " " + next_chunk["content"]
#             
#             # Add the enriched content
#             if additional_content:
#                 enriched_result["content"] = f"{result['content']} {additional_content}"
#             
#             enriched_results.append(enriched_result)
#         
#         logger.info(f"Successfully enriched {len(enriched_results)} results")
#         return enriched_results
#     except Exception as e:
#         error_msg = f"Error in enrich_with_adjacent_chunks: {e}"
#         logger.error(error_msg)
#         logger.error(traceback.format_exc())
#         raise

# Function to generate an answer using LLM with improved prompt
async def generate_answer(question, relevant_results, max_retries=2):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    retries = 0
    while retries < max_retries:    
        try:
            logger.info(f"Generating answer for question: '{question[:50]}...'")
            context = ""
            for result in relevant_results:
                source_type = "Discourse post" if result["source"] == "discourse" else "Documentation"
                context += f"\n\n{source_type} (URL: {result['url']}):\n{result['content'][:1500]}"
            
            # Prepare improved prompt
            prompt = f"""Answer the following question based ONLY on the provided context. 
            If you cannot answer the question based on the context, say "I don't have enough information to answer this question."
            
            Context:
            {context}
            
            Question: {question}
            
            Return your response in this exact format:
            1. A comprehensive yet concise answer
            2. A "Sources:" section that lists the URLs and relevant text snippets you used to answer
            
            Sources must be in this exact format:
            Sources:
            1. URL: [exact_url_1], Text: [brief quote or description]
            2. URL: [exact_url_2], Text: [brief quote or description]
            
            Make sure the URLs are copied exactly from the context without any changes.
            """
            logger.debug(f"LLM prompt length: {len(prompt)} characters")
            logger.info("Sending request to LLM API")
            # Call OpenAI API through aipipe proxy
            url = "https://aipipe.org/openai/v1/chat/completions"
            headers = {
                "Authorization": API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based only on the provided context. Always include sources in your response with exact URLs."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3  # Lower temperature for more deterministic outputs
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received answer from LLM")
                        return result["choices"][0]["message"]["content"]
                    elif response.status == 429:  # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(3 * (retries + 1))  # Exponential backoff
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error generating answer (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except Exception as e:
            error_msg = f"Exception generating answer: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            logger.debug(f"Exception object in generate_answer: {e}")
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(2)  # Wait before retry

# Function to process multimodal content (text + image)
async def process_multimodal_query(question, image_base64):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
        
    try:
        logger.info(f"Processing query: '{question[:50]}...', image provided: {image_base64 is not None}")
        if not image_base64:
            logger.info("No image provided, processing as text-only query")
            return await get_embedding(question)
        
        logger.info("Processing multimodal query with image")
        # Call the GPT-4o Vision API to process the image and question
        url = "https://aipipe.org/openai/v1/chat/completions"
        headers = {
            "Authorization": API_KEY,
            "Content-Type": "application/json"
        }
        
        # Format the image for the API
        image_content = f"data:image/jpeg;base64,{image_base64}"
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Look at this image and tell me what you see related to this question: {question}"},
                        {"type": "image_url", "image_url": {"url": image_content}}
                    ]
                }
            ]
        }
        
        logger.info("Sending request to Vision API")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    image_description = result["choices"][0]["message"]["content"]
                    logger.info(f"Received image description: '{image_description[:50]}...'")
                    
                    # Combine the original question with the image description
                    combined_query = f"{question}\nImage context: {image_description}"
                    
                    # Get embedding for the combined query
                    return await get_embedding(combined_query)
                else:
                    error_text = await response.text()
                    logger.error(f"Error processing image (status {response.status}): {error_text}")
                    # Fall back to text-only query
                    logger.info("Falling back to text-only query")
                    return await get_embedding(question)
    except Exception as e:
        logger.error(f"Exception processing multimodal query: {e}")
        logger.error(traceback.format_exc())
        logger.debug(f"Exception object in process_multimodal_query: {e}")
        # Fall back to text-only query
        logger.info("Falling back to text-only query due to exception")
        return await get_embedding(question)

# Function to parse LLM response and extract answer and sources with improved reliability
def parse_llm_response(response):
    try:
        logger.info("Parsing LLM response")
        
        # First try to split by "Sources:" heading
        parts = response.split("Sources:", 1)
        
        # If that doesn't work, try alternative formats
        if len(parts) == 1:
            # Try other possible headings
            for heading in ["Source:", "References:", "Reference:"]:
                if heading in response:
                    parts = response.split(heading, 1)
                    break
        
        answer = parts[0].strip()
        links = []
        
        if len(parts) > 1:
            sources_text = parts[1].strip()
            source_lines = sources_text.split("\n")
            
            for line in source_lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Remove list markers (1., 2., -, etc.)
                line = re.sub(r'^\d+\.\s*', '', line)
                line = re.sub(r'^-\s*', '', line)
                
                # Extract URL and text using more flexible patterns
                url_match = re.search(r'URL:\s*\[(.*?)\]|url:\s*\[(.*?)\]|\[(http[^\]]+)\]|URL:\s*(http\S+)|url:\s*(http\S+)|(http\S+)', line, re.IGNORECASE)
                text_match = re.search(r'Text:\s*\[(.*?)\]|text:\s*\[(.*?)\]|[""](.*?)[""]|Text:\s*"(.*?)"|text:\s*"(.*?)"', line, re.IGNORECASE)
                
                if url_match:
                    # Find the first non-None group from the regex match
                    url = next((g for g in url_match.groups() if g), "")
                    url = url.strip()
                    
                    # Default text if no match
                    text = "Source reference"
                    
                    # If we found a text match, use it
                    if text_match:
                        # Find the first non-None group from the regex match
                        text_value = next((g for g in text_match.groups() if g), "")
                        if text_value:
                            text = text_value.strip()
                    
                    # Only add if we have a valid URL
                    if url and url.startswith("http"):
                        links.append({"url": url, "text": text})
        
        logger.info(f"Parsed answer (length: {len(answer)}) and {len(links)} sources")
        return {"answer": answer, "links": links}
    except Exception as e:
        error_msg = f"Error parsing LLM response: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        # Return a basic response structure with the error
        return {
            "answer": "Error parsing the response from the language model.",
            "links": []
        }

# Define API routes
@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    try:
        # Log the incoming request
        logger.info(f"Received query request: question='{request.question[:50]}...', image_provided={request.image is not None}")
        logger.debug(f"Full request payload: question='{request.question}', image_present={request.image is not None}")
        
        if not API_KEY:
            error_msg = "API_KEY environment variable not set"
            logger.error(error_msg)
            return JSONResponse(
                status_code=500,
                content={"error": error_msg}
            )
            
        conn = get_db_connection()
        
        try:
            # Process the query (handle text and optional image)
            logger.info("Processing query and generating embedding")
            query_embedding = await process_multimodal_query(
                request.question,
                request.image
            )
            
            # Find similar content
            logger.info("Finding similar content")
            relevant_results = await find_similar_content(query_embedding, conn)
            
            if not relevant_results:
                logger.info("No relevant results found")
                return {
                    "answer": "I couldn't find any relevant information in my knowledge base.",
                    "links": []
                }
            
            # Skip enrichment; use results with fetched content
            enriched_results = relevant_results
            
            # Generate answer
            logger.info("Generating answer")
            llm_response = await generate_answer(request.question, enriched_results)
            
            # Parse the response
            logger.info("Parsing LLM response")
            result = parse_llm_response(llm_response)
            
            # If links extraction failed, create them from the relevant results
            if not result["links"]:
                logger.info("No links extracted, creating from relevant results")
                # Create a dict to deduplicate links from the same source
                links = []
                unique_urls = set()
                
                for res in relevant_results[:5]:  # Use top 5 results
                    url = res["url"]
                    if url not in unique_urls:
                        unique_urls.add(url)
                        snippet = res["content"][:100] + "..." if len(res["content"]) > 100 else res["content"]
                        links.append({"url": url, "text": snippet})
                
                result["links"] = links
            
            # Log the final result structure (now wrapped in a data envelope)
            logger.info(f"Returning result (data envelope): answer_length={len(result['answer'])}, num_links={len(result['links'])}")
            
            # Return the response wrapped in a data envelope
            return result
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            logger.debug(f"Exception object in query_knowledge_base: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": error_msg}
            )
        finally:
            conn.close()
    except Exception as e:
        # Catch any exceptions at the top level
        error_msg = f"Unhandled exception in query_knowledge_base: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        logger.debug(f"Exception object in query_knowledge_base (outer): {e}")
        return JSONResponse(
            status_code=500,
            content={"error": error_msg}
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Try to connect to the database as part of health check
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if tables exist and have data
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        discourse_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        markdown_count = cursor.fetchone()[0]
        
        # Check if any embeddings exist
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
        discourse_embeddings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
        markdown_embeddings = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "status": "healthy", 
            "database": "connected", 
            "api_key_set": bool(API_KEY),
            "discourse_chunks": discourse_count,
            "markdown_chunks": markdown_count,
            "discourse_embeddings": discourse_embeddings,
            "markdown_embeddings": markdown_embeddings
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e), "api_key_set": bool(API_KEY)}
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 