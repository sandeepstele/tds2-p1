# RAG Query API

This is a FastAPI-based backend service that powers a Retrieval-Augmented Generation (RAG) pipeline using a local SQLite knowledge base and Pinecone vector search.

## Features

- Query a multimodal (text + image) knowledge base via `/query` endpoint
- Uses Pinecone for vector similarity search
- GPT-4o-mini model via AIProxy (supports vision for image analysis)
- Auto-generated embeddings using OpenAI-compatible embedding model
- Returns contextual answers along with source links
- Handles rate-limiting and retries on failures
- Health-check endpoint at `/health`

## Requirements

- Python 3.8+
- Environment Variables:
  - `API_KEY` (for AIProxy OpenAI-compatible API)
  - `PINECONE_API_KEY` (your Pinecone API key)
  - `PINECONE_ENV` (Pinecone environment, e.g., `us-east-1-aws`)

## Setup

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with the required keys:
   ```
   API_KEY=your_aipipe_api_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_ENV=us-east-1-aws
   ```

## Running the API

```bash
uvicorn app:app --reload --port 8000
```

## Endpoints

### POST `/query`

Query the knowledge base with text and optionally an image (base64).

**Request**:
```json
{
  "question": "Explain cosine similarity",
  "image": "base64_string_if_any"
}
```

**Response**:
```json
{
  "data": {
    "answer": "Cosine similarity measures ...",
    "links": [
      {"url": "https://example.com", "text": "Relevant source"}
    ]
  }
}
```

### GET `/health`

Checks DB connectivity and whether embeddings exist.

## Database Schema

- `discourse_chunks`: stores forum chunks with embeddings
- `markdown_chunks`: stores markdown doc chunks with embeddings

## Notes

- Handles rate limits and retries gracefully
- Image support via GPT-4o multimodal model
- Logs important events and errors with full tracebacks

## License

MIT
