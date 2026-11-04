from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "CinePulse"
    assert data["engine_ready"] is True


def test_list_users(client):
    response = client.get("/api/users")
    assert response.status_code == 200
    users = response.json()
    assert len(users) >= 4
    usernames = [u["username"] for u in users]
    assert "elena_vance" in usernames
    assert "marcus_sterling" in usernames


def test_recommendations_endpoint(client):
    response = client.get("/api/recommendations/12?k=10&diversity=0.65&novelty=0.15")
    assert response.status_code == 200
    data = response.json()
    assert "hero_featured" in data
    assert "rails" in data
    assert "Top Picks For You" in data["rails"]
    assert len(data["rails"]["Top Picks For You"]) > 0

    top_item = data["rails"]["Top Picks For You"][0]
    assert "title" in top_item
    assert "release_year" in top_item
    assert "rating_mean" in top_item
    assert "final_score" in top_item
    assert "reason" in top_item


def test_feedback_and_realtime_adaptation(client):
    user_idx = 45  # Marcus Sterling
    movie_idx = 10

    # 1. Post a like
    res_like = client.post(
        "/api/feedback",
        json={"user_idx": user_idx, "movie_idx": movie_idx, "action": "like"},
    )
    assert res_like.status_code == 200
    assert res_like.json()["success"] is True

    # 2. Post a dismiss
    dismiss_idx = 20
    res_dismiss = client.post(
        "/api/feedback",
        json={"user_idx": user_idx, "movie_idx": dismiss_idx, "action": "dismiss"},
    )
    assert res_dismiss.status_code == 200

    # 3. Check updated recommendations: dismiss_idx must not appear
    res_recs = client.get(f"/api/recommendations/{user_idx}?k=30")
    assert res_recs.status_code == 200
    all_rec_ids = [item["movie_idx"] for item in res_recs.json()["rails"]["Top Picks For You"]]
    assert dismiss_idx not in all_rec_ids


def test_movie_detail(client):
    response = client.get("/api/movies/50")
    assert response.status_code == 200
    data = response.json()
    assert data["movie_idx"] == 50
    assert "title" in data
    assert "rating_mean" in data
    assert "genres" in data
    assert "similar_movies" in data


def test_search(client):
    response = client.get("/api/search?q=matrix")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0


def test_offline_metrics(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "Two-Tower + LightGBM Ranker" in data["models"]
    assert "Full Pipeline (Retrieval + Ranker + MMR)" in data["models"]
    assert "ndcg@10" in data["models"]["Two-Tower + LightGBM Ranker"]


def test_frontend_serving(client):
    frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if not (frontend_dist / "index.html").exists():
        pytest.skip("Frontend dist not built (tested in frontend build job).")
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "CinePulse" in response.text

