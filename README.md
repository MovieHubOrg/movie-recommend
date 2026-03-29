# Movie Recommendation System

A content-based movie recommendation API built with FastAPI. The system recommends movies based on user watch history using sentence embeddings and Qdrant vector database for efficient similarity search.

## Project Structure

```
movie-recommend/
├── app.py                      # FastAPI application entry point
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
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
│   ├── recommender.py          # recommend_from_history() - Qdrant search
│   ├── user_profiling.py      # build_user_profile() - weighted embeddings
│   └── catalog_service.py    # build/load Qdrant index
│
├── ml/                         # ML components
│   ├── __init__.py
│   ├── model_loader.py        # SentenceTransformer (GPU/CPU)
│   └── embeddings.py          # Embedding generation utilities
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── text.py                # HTML parsing, content string builder
│   └── engagement.py         # Engagement score calculation
│
├── core/                      # Configuration
│   ├── __init__.py
│   └── config.py            # Settings from environment variables
│
├── qdrant_data/               # Qdrant vector database (auto-created)
│
└── tests/                     # Test suite
    └── __init__.py
```

## How It Works

The system recommends movies based on user watch history using content-based filtering:

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
4. **Catalog Index**: Builds Qdrant collection with cosine distance for similarity search
5. **Recommendation**: Searches Qdrant for top-k similar movies, excludes already watched

## External API Integration

The system integrates with an external movie API for fetching movie catalog and user watch history:

- `MOVIE_API`: Base URL of the external movie API
- `Authorization` header (Bearer token) for authenticated endpoints

## Setup

**Requires Python 3.11**

### Create Virtual Environment

```bash
py -3.11 -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
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
pydantic
requests
python-dotenv
xmltodict
pytest
qdrant-client
```

## Configuration

Environment variables (optional, defaults in `core/config.py`):

| Variable            | Default             | Description                     |
| -------------------| -------------------| ------------------------------ |
| `APP_NAME`          | Movie Recommend API | Application title              |
| `DEBUG`            | False              | Enable debug mode               |
| `MODEL_NAME`       | all-MiniLM-L6-v2  | Sentence transformer model      |
| `CACHE_FOLDER`     | ./models           | Model cache directory            |
| `QDRANT_PATH`      | ./qdrant_data     | Qdrant database path            |
| `DEFAULT_TOP_K`   | 5                 | Default recommendations count    |
| `SEARCH_MULTIPLIER`| 3                 | Search multiplier             |
| `USE_GPU`          | True              | Enable GPU acceleration       |
| `MOVIE_API`        | -                 | External movie API URL       |

## Run the Project

```bash
uvicorn app:app --reload
```

Server: `http://127.0.0.1:8000`  
Docs: `http://127.0.0.1:8000/docs`

## API Endpoints

| Endpoint                  | Method | Description                      |
| ------------------------- | ------ | -------------------------------- |
| `/`                       | GET    | Health check                     |
| `/api/v1/movie/list`      | GET    | Get movie list from external API|
| `/api/v1/movie/history`  | GET    | Get watch history (Bearer token)   |
| `/api/v1/movie/recommend` | GET    | Get recommendations                |

### Example

```bash
# Get recommendations (requires Bearer token)
curl -H "Authorization: Bearer YOUR_TOKEN" "http://127.0.0.1:8000/api/v1/movie/recommend?top_k=5"
```

### Response

```json
{
  "result": true,
  "message": "Get recommend list successfully",
  "data": [
    {
      "id": 8961782939090944,
      "title": "Movie Title",
      "similarity": 0.8723
    }
  ]
}
```

## Architecture

```
Client ──▶ FastAPI ──▶ Services ──▶ Qdrant
              │           │
              │           ├─▶ ML (SentenceTransformer)
              │           └─▶ Utils (text, engagement)
              │
              └─▶ External Movie API
```

## Key Components

### Services Layer
- **recommender.py**: Qdrant vector search for similar movies
- **user_profiling.py**: Weighted user profile from watch history
- **catalog_service.py**: Qdrant index management

### ML Layer
- **model_loader.py**: SentenceTransformer loader (GPU/CPU)
- **embeddings.py**: Movie embedding generation

### Utils
- **text.py**: HTML parsing, content string builder
- **engagement.py**: Engagement score calculation