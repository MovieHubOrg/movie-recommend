from mock_data import movie_list, movie_history_list
from recommend import build_user_profile, recommend_from_history

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello():
    return {"message": "Hello World"}

@app.get("/movies")
def get_movies():
    return movie_list

@app.get("/movie-histories")
def get_movie_histories():
    return movie_history_list

@app.get("/movie/recommend")
def get_movie_recommend():

    history = movie_history_list["data"]
    print("=== Engagement scores ===")
    user_vec = build_user_profile(history)
    print(f"\nUser profile vector shape: {user_vec.shape}")
    print(f"Sample dims: {user_vec[:5].round(4)}")

    catalog = movie_list["data"]["content"]
    history = movie_history_list["data"]

    user_vec = build_user_profile(history)
    recs = recommend_from_history(user_vec, catalog, history)

    return {"data": recs}