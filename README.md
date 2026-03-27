# Movie Recommendation System

A content-based movie recommendation API built with FastAPI. The system recommends movies based on user watch history using sentence embeddings and cosine similarity.

## Project Structure

| File | Description |
|------|-------------|
| `app.py` | FastAPI endpoints - provides `/movies`, `/movie-histories`, and `/movie/recommend` APIs |
| `recommend.py` | Core recommendation logic - builds user profile from watch history and generates recommendations |
| `get_catalog_embeddings.py` | Embedding caching for catalog movies using Sentence Transformers |
| `util.py` | Utilities - HTML parsing, content string building, engagement score calculation |
| `mock_data.py` | Sample movie catalog and watch history data |

## How It Works

1. **User Profile Building**: Extracts movie metadata from watch history, converts to text, and generates embeddings using `all-MiniLM-L6-v2`
2. **Engagement Score**: Calculates user preference weight based on:
   - `timesWatched` - rewatch count (weight: 0.5)
   - `lastWatchSeconds / duration` - completion percentage (weight: 1.0)
   - `modifiedDate` - recency (weight: 0.3)
3. **Recommendation**: Computes cosine similarity between user profile vector and catalog embeddings, returns top-k similar movies

## Setup

### Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
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
```

Install with: `pip install -r requirements.txt`

## Run the Project

```bash
uvicorn app:app --reload
```

Server runs at: http://127.0.0.1:8000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Hello world |
| `/movies` | GET | Get all movies in catalog |
| `/movie-histories` | GET | Get user watch history |
| `/movie/recommend` | GET | Get movie recommendations based on history |

## Example Output

`GET /movie/recommend` returns top 5 recommended movies with similarity scores:

```json
{
  "data": [
    {"id": 8961782939090944, "title": "Kaiju No. 8", "similarity": 0.8723, ...},
    {"id": 8998635757764608, "title": "Tôi thăng cấp một mình", "similarity": 0.8451, ...}
  ]
}
```