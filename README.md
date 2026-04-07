# Movie Recommendation System

A content-based movie recommendation API built with **FastAPI**, using **Sentence Transformer** embeddings and **Qdrant** vector database for efficient similarity search.

## Overview

Given a list of movie IDs, the system returns the most similar movies based on content metadata (title, description, genres, countries, year). It works by:

1. **Startup** — fetches the full movie catalog from an external API and builds a Qdrant vector index. A hash of the catalog is saved on disk; if the catalog changes on next startup, the index is automatically rebuilt.
2. **Embedding generation** — each movie's metadata is combined into a single text string and embedded via `all-MiniLM-L6-v2`.
3. **Recommendation** — vectors of the input movie IDs are retrieved, averaged into a query vector, and used to search Qdrant for the most similar movies (excluding the inputs themselves).

## Architecture

```
Client ──▶ FastAPI ──▶ Services ──▶ Qdrant Vector DB
              │           │
              │           └─▶ SentenceTransformer embeddings
              │
              └─▶ External Movie API (startup only)
```

## Project Structure

```
movie-recommend/
├── app.py                     # FastAPI entry point + lifespan
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
│
├── api/routes/
│   └── movie.py               # /recommend-by-movies endpoint
│
├── services/
│   ├── recommender.py          # Vector search & averaging logic
│   └── catalog_service.py      # Catalog index build / load
│
├── ml/
│   ├── model_loader.py         # SentenceTransformer (auto GPU/CPU)
│   └── embeddings.py           # Embedding utilities
│
├── utils/
│   └── text.py                 # HTML parsing, content builder
│
├── core/
│   └── config.py               # Settings from env vars
│
├── qdrant_data/               # Qdrant vector database (auto-created)
│
└── tests/                     # Test suite
```

## Setup

**Requires Python 3.11**

```bash
py -3.11 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### Configuration

Environment variables (defaults in `core/config.py`):

| Variable            | Default             | Description                   |
| ------------------- | ------------------- | ----------------------------- |
| `APP_NAME`          | Movie Recommend API | Application title             |
| `DEBUG`             | False               | Enable debug mode             |
| `MODEL_NAME`        | all-MiniLM-L6-v2    | Sentence transformer model    |
| `CACHE_FOLDER`      | ./models            | Model cache directory         |
| `QDRANT_PATH`       | ./qdrant_data       | Qdrant database path          |
| `DEFAULT_TOP_K`     | 5                   | Default recommendations count |
| `SEARCH_MULTIPLIER` | 3                   | Search multiplier             |
| `MOVIE_API`         | -                   | External movie API URL        |

## Run

```bash
uvicorn app:app --reload
```

Server: `http://127.0.0.1:8000`
Swagger docs: `http://127.0.0.1:8000/docs`

## API Endpoints

| Endpoint                                     | Method | Description                      |
| -------------------------------------------- | ------ | -------------------------------- |
| `/`                                          | GET    | Health check                     |
| `/api/v1/movie/recommend-by-movies`          | GET    | Get recommendations by movie IDs |

### Recommend by Movie IDs

```bash
curl "http://127.0.0.1:8000/api/v1/movie/recommend-by-movies?movieIds=1,2,3&top_k=5"
```

**Query Parameters:**

| Parameter  | Type   | Required | Default | Description               |
| ---------- | ------ | -------- | ------- | ------------------------- |
| `movieIds` | string | Yes      | -       | Comma-separated movie IDs |
| `top_k`    | int    | No       | 5       | Number of recommendations |

**Response:**

```json
{
  "result": true,
  "message": "Get recommend list successfully",
  "data": ["8961782939090944", "1234567890123456", "9876543210987654"]
}
```

## Tests

```bash
pytest tests/
```
