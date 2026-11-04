import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_action_fan_persona_coherence(client):
    """
    User 45 (Marcus Sterling) is an Action/Sci-Fi enthusiast.
    Verifies that top recommendations contain Action, Thriller, or Sci-Fi genres.
    """
    response = client.get("/api/recommendations/45?k=20")
    assert response.status_code == 200
    data = response.json()
    assert "rails" in data
    top_picks = data["rails"]["Top Picks For You"]
    assert len(top_picks) > 0

    action_scifi_count = sum(
        1 for item in top_picks[:10]
        if any(g in ["Action", "Sci-Fi", "Thriller", "Adventure"] for g in item["genres"])
    )
    assert action_scifi_count >= 5, f"Expected at least 5/10 Action/SciFi/Thriller recommendations, got {action_scifi_count}"


def test_scifi_fan_persona_coherence(client):
    """
    User 12 (Elena Vance) is a Sci-Fi/Noir enthusiast.
    Verifies that recommendations contain Sci-Fi, Crime, Mystery, or Thriller.
    """
    response = client.get("/api/recommendations/12?k=20")
    assert response.status_code == 200
    data = response.json()
    top_picks = data["rails"]["Top Picks For You"]
    assert len(top_picks) > 0

    scifi_noir_count = sum(
        1 for item in top_picks[:10]
        if any(g in ["Sci-Fi", "Crime", "Mystery", "Film-Noir", "Thriller"] for g in item["genres"])
    )
    assert scifi_noir_count >= 4, f"Expected at least 4/10 SciFi/Crime/Mystery recommendations, got {scifi_noir_count}"


def test_cold_start_onboarding_persona(client):
    """
    Verifies that a new user who onboards with Comedy and Romance preferences
    receives recommendations aligned with those genres.
    """
    # Record onboarding preferences via user creation
    unique_user = f"chloe_{uuid.uuid4().hex[:8]}"
    res = client.post("/api/users", json={
        "username": unique_user,
        "display_name": "Chloe Bennett",
        "preferred_genres": ["Comedy", "Romance"],
    })
    assert res.status_code == 200
    new_user_idx = res.json()["user_idx"]

    response = client.get(f"/api/recommendations/{new_user_idx}?k=20")
    assert response.status_code == 200
    data = response.json()
    top_picks = data["rails"]["Top Picks For You"]
    assert len(top_picks) > 0

    matching_genres = sum(
        1 for item in top_picks[:10]
        if any(g in ["Comedy", "Romance", "Drama"] for g in item["genres"])
    )
    assert matching_genres >= 6, f"Expected at least 6/10 matching onboarding preferences, got {matching_genres}"


def test_hero_quality_gate_never_picks_poor_movies(client):
    """
    Verifies the Hero Spotlight quality gate:
    Sub-3.1★ or low-confidence movies must NEVER be selected as the Hero Spotlight.
    Enforces bayesian_rating >= 3.20, rating_mean >= 3.10, and rating_count >= 50.
    """
    # Test across multiple users
    for uid in [12, 45, 128, 256, 500]:
        response = client.get(f"/api/recommendations/{uid}?k=10")
        assert response.status_code == 200
        data = response.json()
        hero = data.get("hero_featured")
        if hero:
            assert hero["rating_mean"] >= 3.10, (
                f"Hero movie '{hero['title']}' has low community rating: {hero['rating_mean']} < 3.10"
            )
            assert hero["bayesian_rating"] >= 3.20, (
                f"Hero movie '{hero['title']}' has low bayesian rating: {hero['bayesian_rating']} < 3.20"
            )
            assert hero["rating_count"] >= 50, (
                f"Hero movie '{hero['title']}' has low review count confidence: {hero['rating_count']} < 50"
            )


def test_dislike_suppression(client):
    """
    Verifies that disliking a movie immediately removes it from recommendations.
    """
    uid = 45
    feed_before = client.get(f"/api/recommendations/{uid}?k=15").json()
    top_picks = feed_before["rails"]["Top Picks For You"]
    target_movie = top_picks[0]
    target_idx = target_movie["movie_idx"]

    # Dislike the top movie
    dislike_res = client.post("/api/feedback", json={
        "user_idx": uid,
        "movie_idx": target_idx,
        "action": "dislike",
    })
    assert dislike_res.status_code == 200

    # Next recommendation feed must NOT contain this movie
    feed_after = client.get(f"/api/recommendations/{uid}?k=20").json()
    after_movie_ids = [m["movie_idx"] for m in feed_after["rails"]["Top Picks For You"]]
    assert target_idx not in after_movie_ids, f"Movie {target_idx} was disliked but still appeared in recommendations!"


def test_affinity_labels_and_community_ratings(client):
    """
    Verifies transparent affinity labeling (replacing deceptive pseudo-probabilities)
    and confirms community ratings are cleanly separated from personalized scores.
    """
    response = client.get("/api/recommendations/12?k=20")
    data = response.json()
    top_picks = data["rails"]["Top Picks For You"]

    valid_labels = {"Top Affinity Pick", "High Affinity", "Curated Recommendation"}
    for item in top_picks:
        assert item.get("affinity_label") in valid_labels, f"Invalid affinity label: {item.get('affinity_label')}"
        assert "rating_mean" in item, "Community rating missing"
        assert item["rating_mean"] >= 0.0, "Invalid community rating"
        assert "final_score" in item, "Personalized model score missing"
