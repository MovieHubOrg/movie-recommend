# Movie Recommendation System

A content-based movie recommendation API built with FastAPI. The system recommends movies based on user watch history using sentence embeddings and FAISS for efficient similarity search.

## Project Structure

| File               | Description                                                                             |
| ------------------ | --------------------------------------------------------------------------------------- |
| `app.py`           | FastAPI endpoints - provides `/movies`, `/movie-histories`, and `/movie/recommend` APIs |
| `model_loader.py`  | Loads Sentence Transformer model with GPU/CPU support                                   |
| `user_profile.py`  | Builds user profile vector from watch history using weighted embeddings                 |
| `catalog_index.py` | Builds and loads FAISS index for fast similarity search on catalog                      |
| `recommender.py`   | Generates recommendations using FAISS index search                                      |
| `util.py`          | Utilities - HTML parsing, content string building, engagement score calculation         |
| `mock_data.py`     | Sample movie catalog and watch history data                                             |

## How It Works

1. **Model Loading**: Loads `all-MiniLM-L6-v2` model with automatic GPU/CPU detection
2. **User Profile Building**:
   - Extracts movie metadata from watch history
   - Converts to text content string
   - Generates embeddings using the model
   - Computes weighted average based on engagement score
3. **Engagement Score**: Calculates user preference weight based on:
   - `timesWatched` - rewatch count (weight: 0.5)
   - `lastWatchSeconds / duration` - completion percentage (weight: 1.0)
   - `modifiedDate` - recency (weight: 0.3)
4. **Catalog Index**: Pre-builds FAISS index (FlatIP) for fast inner product search
5. **Recommendation**: Searches FAISS index for top-k similar movies, excludes already watched

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
```

## Run the Project

```bash
uvicorn app:app --reload
```

Server runs at: http://127.0.0.1:8000

## API Endpoints

| Endpoint           | Method | Description                                |
| ------------------ | ------ | ------------------------------------------ |
| `/`                | GET    | Hello world                                |
| `/movies`          | GET    | Get all movies in catalog                  |
| `/movie-histories` | GET    | Get user watch history                     |
| `/movie/recommend` | GET    | Get movie recommendations based on history |

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
