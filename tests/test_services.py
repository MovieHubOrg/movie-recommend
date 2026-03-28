import pytest
from datetime import datetime, timedelta
import json
from unittest.mock import Mock, patch
from fastapi import Response

from utils.text import parse_html, build_content_string
from utils.engagement import compute_engagement_score
from services.user_profiling import build_user_profile
from services.recommender import recommend_from_history
from api.routes.movie import parse_response, get_movie_client


class TestTextUtils:
    def test_parse_html(self):
        html = "<p>Hello <b>World</b></p>"
        result = parse_html(html)
        assert "Hello" in result
        assert "World" in result

    def test_parse_html_empty(self):
        assert parse_html("") == ""
        assert parse_html("<div></div>") == ""

    def test_parse_html_with_special_chars(self):
        html = "<p>Test &amp; &lt; &gt; quotes</p>"
        result = parse_html(html)
        assert "Test" in result

    def test_build_content_string(self):
        movie = {
            "title": "Test Movie",
            "originalTitle": "Original",
            "description": "<p>Description</p>",
            "categories": [{"name": "Action"}, {"name": "Comedy"}],
            "country": "USA",
            "year": 2024
        }
        result = build_content_string(movie)
        assert "Test Movie" in result
        assert "Action" in result
        assert "Comedy" in result
        assert "USA" in result
        assert "2024" in result

    def test_build_content_string_empty(self):
        movie = {}
        result = build_content_string(movie)
        assert result.strip() == ""

    def test_build_content_string_partial(self):
        movie = {"title": "Only Title"}
        result = build_content_string(movie)
        assert "Only Title" in result


class TestEngagement:
    def test_engagement_full_completion(self):
        history = {
            "movie": {"metadata": json.dumps({"duration": 100})},
            "timesWatched": 2,
            "lastWatchSeconds": 100,
            "modifiedDate": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
        score = compute_engagement_score(history)
        assert score > 0

    def test_engagement_partial_completion(self):
        history = {
            "movie": {"metadata": json.dumps({"duration": 100})},
            "timesWatched": 1,
            "lastWatchSeconds": 50,
            "modifiedDate": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
        score = compute_engagement_score(history)
        assert 0 < score < 1.5

    def test_engagement_no_metadata(self):
        history = {
            "movie": {},
            "timesWatched": 0,
            "lastWatchSeconds": 0,
            "modifiedDate": ""
        }
        score = compute_engagement_score(history)
        assert score > 0

    def test_engagement_invalid_date(self):
        history = {
            "movie": {},
            "timesWatched": 1,
            "lastWatchSeconds": 50,
            "modifiedDate": "invalid-date"
        }
        score = compute_engagement_score(history)
        assert score > 0

    def test_engagement_recency_decay(self):
        old_date = (datetime.now() - timedelta(days=60)).strftime("%d/%m/%Y %H:%M:%S")
        history = {
            "movie": {"metadata": json.dumps({"duration": 100})},
            "timesWatched": 0,
            "lastWatchSeconds": 100,
            "modifiedDate": old_date
        }
        old_score = compute_engagement_score(history)

        recent_date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        history["modifiedDate"] = recent_date
        recent_score = compute_engagement_score(history)

        assert recent_score > old_score

    def test_engagement_zero_duration(self):
        history = {
            "movie": {"metadata": json.dumps({"duration": 0})},
            "timesWatched": 1,
            "lastWatchSeconds": 100,
            "modifiedDate": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
        score = compute_engagement_score(history)
        assert score > 0

    def test_engagement_high_times_watched(self):
        history = {
            "movie": {"metadata": json.dumps({"duration": 100})},
            "timesWatched": 10,
            "lastWatchSeconds": 100,
            "modifiedDate": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
        score = compute_engagement_score(history)
        assert score > 5


class TestUserProfiling:
    def test_empty_history(self):
        result = build_user_profile([])
        assert result is None

    def test_history_with_items(self):
        history = [
            {
                "movie": {
                    "title": "Movie 1",
                    "originalTitle": "",
                    "description": "Action movie",
                    "categories": [{"name": "Action"}],
                    "country": "USA",
                    "year": 2024
                },
                "timesWatched": 2,
                "lastWatchSeconds": 100,
                "modifiedDate": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            }
        ]
        result = build_user_profile(history)
        assert result is not None
        assert len(result) > 0

    def test_history_multiple_movies(self):
        history = [
            {
                "movie": {
                    "title": "Movie 1",
                    "originalTitle": "",
                    "description": "Action movie",
                    "categories": [{"name": "Action"}],
                    "country": "USA",
                    "year": 2024
                },
                "timesWatched": 1,
                "lastWatchSeconds": 50,
                "modifiedDate": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            },
            {
                "movie": {
                    "title": "Movie 2",
                    "originalTitle": "",
                    "description": "Comedy movie",
                    "categories": [{"name": "Comedy"}],
                    "country": "UK",
                    "year": 2023
                },
                "timesWatched": 3,
                "lastWatchSeconds": 90,
                "modifiedDate": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            }
        ]
        result = build_user_profile(history)
        assert result is not None
        assert len(result) > 0


class TestRecommender:
    def test_none_user_vec(self):
        result = recommend_from_history(None, [], [])
        assert result == []

    def test_empty_catalog(self):
        import numpy as np
        user_vec = np.zeros(384)
        result = recommend_from_history(user_vec, [], [])
        assert result == []

    def test_excludes_watched_movies(self):
        import numpy as np
        user_vec = np.zeros(384)
        catalog = [
            {"id": 1, "title": "Movie 1"},
            {"id": 2, "title": "Movie 2"},
            {"id": 3, "title": "Movie 3"}
        ]
        history = [
            {"movie": {"id": 1}},
            {"movie": {"id": 2}}
        ]
        result = recommend_from_history(user_vec, catalog, history, top_k=5)
        watched_ids = {item["movie"]["id"] for item in history}
        for movie in result:
            assert movie["id"] not in watched_ids


class TestMovieRoutes:
    def test_parse_response_json(self):
        response = Mock()
        response.headers = {"content-type": "application/json"}
        response.content = b'{"key": "value"}'
        response.status_code = 200
        
        result = parse_response(response)
        assert result.status_code == 200

    def test_parse_response_xml_error(self):
        response = Mock()
        response.headers = {"content-type": "application/xml"}
        response.content = b"<error>Not found</error>"
        response.status_code = 404
        
        result = parse_response(response)
        assert result.status_code == 404

    def test_parse_response_xml_prefix(self):
        response = Mock()
        response.headers = {}
        response.content = b"<error>Not found</error>"
        response.status_code = 400
        
        result = parse_response(response)
        assert result.status_code == 400

    @patch('api.routes.movie.settings')
    def test_get_movie_client_without_token(self, mock_settings):
        mock_settings.movie_api = "http://test.com"
        client_class = get_movie_client()
        client = client_class()
        assert client.token == ""
        assert client.base_url == "http://test.com"

    @patch('api.routes.movie.settings')
    def test_get_movie_client_with_bearer_token(self, mock_settings):
        mock_settings.movie_api = "http://test.com"
        client_class = get_movie_client("Bearer mytoken")
        client = client_class()
        assert client.token == "mytoken"

    @patch('api.routes.movie.settings')
    def test_get_movie_client_with_invalid_bearer(self, mock_settings):
        mock_settings.movie_api = "http://test.com"
        client_class = get_movie_client("Basic token")
        client = client_class()
        assert client.token == ""


class TestEdgeCases:
    def test_engagement_corrupted_metadata(self):
        history = {
            "movie": {"metadata": "not valid json"},
            "timesWatched": 1,
            "lastWatchSeconds": 50,
            "modifiedDate": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
        score = compute_engagement_score(history)
        assert score > 0

    def test_engagement_completion_exceeds_duration(self):
        history = {
            "movie": {"metadata": json.dumps({"duration": 50})},
            "timesWatched": 0,
            "lastWatchSeconds": 100,
            "modifiedDate": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
        score = compute_engagement_score(history)
        assert score > 0
