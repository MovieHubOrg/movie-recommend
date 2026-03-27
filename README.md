# Movie Recommendation System

A content-based movie recommendation API built with FastAPI. The system recommends movies based on user watch history using sentence embeddings and FAISS for efficient similarity search.

## Project Structure

```
movie-recommend/
├── app.py                      # FastAPI application entry point
├── mock_data.py                # Sample movie catalog and watch history
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── api/                        # API layer (routes & schemas)
│   ├── __init__.py
│   ├── schemas.py              # Pydantic models for request/response
│   └── routes/
│       ├── __init__.py         # Main router (combines sub-routers)
│       ├── movies.py           # GET /api/v1/movies endpoints
│       └── recommendations.py  # GET /api/v1/movie/recommend
│
├── services/                   # Business logic layer
│   ├── __init__.py
│   ├── recommender.py          # recommend_from_history() - FAISS search
│   ├── user_profiling.py       # build_user_profile() - weighted embeddings
│   └── catalog_service.py      # build/load FAISS index & cache
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
```

## Configuration

Environment variables (optional, defaults in `core/config.py`):

| Variable            | Default                     | Description                       |
| ------------------- | --------------------------- | --------------------------------- |
| `APP_NAME`          | Movie Recommend API         | Application title                 |
| `DEBUG`             | False                       | Enable debug mode                 |
| `MODEL_NAME`        | all-MiniLM-L6-v2            | Sentence transformer model        |
| `CACHE_FOLDER`      | ./models                    | Model cache directory             |
| `INDEX_FILE`        | ./models/catalog_faiss.index| FAISS index file path             |
| `EMBEDDINGS_CACHE`  | ./models/catalog_embeddings.npz | Cached embeddings file       |
| `DEFAULT_TOP_K`     | 5                           | Default number of recommendations |
| `SEARCH_MULTIPLIER` | 3                           | Search multiplier for retrieval   |
| `USE_GPU`           | True                        | Enable GPU acceleration           |

## Run the Project

```bash
uvicorn app:app --reload
```

Server runs at: `http://127.0.0.1:8000`

API documentation: `http://127.0.0.1:8000/docs`

## API Endpoints

| Endpoint                   | Method | Description                          |
| -------------------------- | ------ | ------------------------------------ |
| `/`                        | GET    | Health check                         |
| `/api/v1/movies`           | GET    | Get all movies in catalog            |
| `/api/v1/movies/histories` | GET    | Get user watch history               |
| `/api/v1/movie/recommend`  | GET    | Get recommendations (param: `top_k`) |

### Example Request

```bash
curl "http://127.0.0.1:8000/api/v1/movie/recommend?top_k=5"
```

### Example Response

```json
{
  "data": [
    {
      "id": 8961782939090944,
      "title": "Kaiju No. 8",
      "similarity": 0.8723,
      ...
    },
    {
      "id": 8998635757764608,
      "title": "Tôi thăng cấp một mình",
      "similarity": 0.8451,
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
