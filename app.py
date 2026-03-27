from mock_data import movie_list, movie_history_list
from recommender import recommend_from_history
from user_profile import build_user_profile

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
    catalog = movie_list["data"]["content"]

    user_vec = build_user_profile(history)
    recs = recommend_from_history(user_vec, catalog, history, top_k=5)

    return {"data": recs}