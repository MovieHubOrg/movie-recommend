# Movie Recommendation System

A content-based movie recommendation API built with FastAPI. The system recommends movies based on user watch history using sentence embeddings and FAISS for efficient similarity search.

## Project Structure

```
movie-recommend/
├── app.py                      # FastAPI application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── api/                        # API layer
│   ├── __init__.py
│   └── routes/
│       ├── __init__.py         # Main router
│       └── movie.py            # Movie & recommendation endpoints
│
├── services/                   # Business logic layer
│   ├── __init__.py
│   ├── recommender.py          # recommend_from_history() - FAISS search
│   ├── user_profiling.py       # build_user_profile() - weighted embeddings
│   └── catalog_service.py     # build/load FAISS index & cache
│
├── ml/                         # ML components
│   ├── __init__.py
│   ├── model_loader.py         # SentenceTransformer (GPU/CPU)
│   └── embeddings.py           # Embedding generation utilities
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── text.py                 # HTML parsing, content string builder
│   └── engagement.py           # Engagement score calculation
│
├── core/                       # Configuration
│   ├── __init__.py
│   └── config.py               # Settings from environment variables
│
├── models/                     # ML artifacts (gitignored)
│   ├── catalog_embeddings.npz  # Cached movie embeddings
│   └── catalog_faiss.index     # FAISS index for similarity search
│
└── tests/                      # Test suite
    └── __init__.py
```

## How It Works

1. **Model Loading**: Loads `all-MiniLM-L6-v2` sentence transformer with GPU/CPU auto-detection
2. **User Profile Building**:
   - Extracts movie metadata (title, description, categories, country, year)
   - Converts to text content string
   - Generates embeddings using the model
   - Computes weighted average based on engagement scores
3. **Engagement Score**: Calculates user preference weight based on:
   - `timesWatched` - rewatch count (weight: 0.5)
   - `lastWatchSeconds / duration` - completion percentage (weight: 1.0)
   - `modifiedDate` - recency decay (weight: 0.3)
4. **Catalog Index**: Pre-builds FAISS index (FlatIP) for fast inner product search
5. **Recommendation**: Searches FAISS index for top-k similar movies, excludes watched
6. **External API Integration**: The system proxies requests to an external movie API, handling authentication via Bearer tokens and converting XML error responses to JSON format

## External API Integration

The system integrates with an external movie API for fetching movie catalog and user watch history. The following environment variable must be configured:

- `MOVIE_API`: Base URL of the external movie API

Optionally, you can pass a Bearer token via the `Authorization` header for authenticated endpoints.

When the API returns a 401 Unauthorized error, the response is:

```json
{
  "result": false,
  "message": "Missing Authorization header"
}
```

## Setup

**Requires Python 3.11**

### Create Virtual Environment

```bash
# Create venv
py -3.11 -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

## Requirements

```
fastapi
uvicorn
numpy
scikit-learn
sentence-transformers
beautifulsoup4
torch
faiss-cpu
pydantic
requests
python-dotenv
xmltodict
```

## Configuration

Environment variables (optional, defaults in `core/config.py`):

| Variable              | Default                     | Description                       |
| --------------------- | --------------------------- | --------------------------------- |
| `APP_NAME`            | Movie Recommend API        | Application title                 |
| `DEBUG`               | False                       | Enable debug mode                 |
| `MODEL_NAME`          | all-MiniLM-L6-v2            | Sentence transformer model        |
| `CACHE_FOLDER`        | ./models                    | Model cache directory             |
| `INDEX_FILE`          | ./models/catalog_faiss.index| FAISS index file path             |
| `EMBEDDINGS_CACHE`   | ./models/catalog_embeddings.npz | Cached embeddings file       |
| `DEFAULT_TOP_K`       | 5                           | Default number of recommendations |
| `SEARCH_MULTIPLIER`   | 3                           | Search multiplier for retrieval   |
| `USE_GPU`             | True                        | Enable GPU acceleration           |
| `MOVIE_API`           | -                           | External movie API base URL       |

## Run the Project

```bash
uvicorn app:app --reload
```

Server runs at: `http://127.0.0.1:8000`

API documentation: `http://127.0.0.1:8000/docs`

## API Endpoints

| Endpoint                        | Method | Description                          |
| ------------------------------- | ------ | ------------------------------------ |
| `/`                             | GET    | Health check                         |
| `/api/v1/movie/list`           | GET    | Get movie list from external API    |
| `/api/v1/movie/history`        | GET    | Get user watch history (Bearer token) |
| `/api/v1/movie/recommend`      | GET    | Get recommendations (param: `top_k`) |

### Example Request

```bash
# Get movie recommendations (requires Bearer token)
curl -H "Authorization: Bearer YOUR_TOKEN" "http://127.0.0.1:8000/api/v1/movie/recommend?top_k=5"

# Get movie list from external API
curl "http://127.0.0.1:8000/api/v1/movie/list"

# Get watch history from external API (requires Bearer token)
curl -H "Authorization: Bearer YOUR_TOKEN" "http://127.0.0.1:8000/api/v1/movie/history"
```

### Example Response

```json
{
  "result": true,
  "message": "Get recommend list successfully",
  "data": [
    {
      "id": 8961782939090944,
      "title": "Kaiju No. 8",
      "similarity": 0.8723,
      ...
    }
  ]
}
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│  FastAPI     │────▶│  Services   │
│             │     │  (app.py)    │     │  Layer      │
└─────────────┘     └──────────────┘     └─────────────┘
                           │                    │
                           │                    ▼
                           │            ┌─────────────┐
                           │            │   ML        │
                           │            │  (model)    │
                           │            └─────────────┘
                           │                    │
                           ▼                    ▼
                     ┌──────────────┐    ┌─────────────┐
                     │   Utils      │    │   FAISS     │
                     │   (text,     │    │   Index     │
                     │   engagement)│    │             │
                     └──────────────┘    └─────────────┘
```
