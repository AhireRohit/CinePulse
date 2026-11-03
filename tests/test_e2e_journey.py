from fastapi.testclient import TestClient

from src.api.main import app


def test_complete_user_lifecycle_journey():
    """
    End-to-End User Lifecycle & Session Personalization Verification:
    1. System health check & model artifact verification.
    2. Profile selection (Elena Vance - Sci-Fi fan).
    3. Multi-stage personalized candidate retrieval, ranking, and MMR reranking.
    4. Real-time feedback loop: User likes an item and dismisses another.
    5. Instant adaptation check: Dismissed item is suppressed, liked item is registered.
    6. Cold-start onboarding journey: Create a new user with zero prior dataset history,
       calibrate preference vector via genre multi-hot projection, verify recommendations.
    7. Recommendation Explainability & Diagnostics check.
    8. Empirical offline benchmark metrics contract.
    """
    with TestClient(app) as client:
        # Step 1: Health & Readiness Check
        res_health = client.get("/health")
        assert res_health.status_code == 200
        health_data = res_health.json()
        assert health_data["status"] == "healthy"
        assert health_data["engine_ready"] is True

        # Step 2: List profiles
        res_users = client.get("/api/users")
        assert res_users.status_code == 200
        users = res_users.json()
        assert len(users) >= 4
        elena = next(u for u in users if u["username"] == "elena_vance")

        # Step 3: Fetch Elena's personalized feed
        res_feed = client.get(f"/api/recommendations/{elena['user_idx']}?k=15&diversity=0.65")
        assert res_feed.status_code == 200
        feed_data = res_feed.json()
        assert feed_data["hero_featured"] is not None
        assert "Top Picks For You" in feed_data["rails"]
        assert len(feed_data["rails"]["Top Picks For You"]) > 0
        assert feed_data["timings_ms"]["total_ms"] > 0

        # Step 4: Real-time feedback loop
        top_movie = feed_data["rails"]["Top Picks For You"][0]
        second_movie = feed_data["rails"]["Top Picks For You"][1]

        # 4a. Like top movie
        res_like = client.post(
            "/api/feedback",
            json={"user_idx": elena["user_idx"], "movie_idx": top_movie["movie_idx"], "action": "like"},
        )
        assert res_like.status_code == 200

        # 4b. Dismiss second movie
        res_dismiss = client.post(
            "/api/feedback",
            json={"user_idx": elena["user_idx"], "movie_idx": second_movie["movie_idx"], "action": "dismiss"},
        )
        assert res_dismiss.status_code == 200

        # Step 5: Verify immediate session adaptation
        res_feed_after = client.get(f"/api/recommendations/{elena['user_idx']}?k=25")
        assert res_feed_after.status_code == 200
        updated_items = res_feed_after.json()["rails"]["Top Picks For You"]
        updated_indices = [item["movie_idx"] for item in updated_items]

        # Dismissed item MUST be completely absent from future recommendation lists
        assert second_movie["movie_idx"] not in updated_indices, "Dismissed item was not suppressed!"

        # Step 6: Cold-Start Onboarding Flow
        import time
        ts = int(time.time())
        new_profile_payload = {
            "username": f"alexandra_noir_{ts}",
            "display_name": "Alexandra Noir",
            "preferred_genres": ["Film-Noir", "Mystery", "Crime"],
            "favorite_movies": [50, 846],  # Seed items
            "avatar_color": "#D4AF37",
            "bio": "Cold-start evaluation user specializing in vintage mystery cinema.",
        }
        res_create = client.post("/api/users", json=new_profile_payload)
        assert res_create.status_code == 200
        new_user = res_create.json()
        assert new_user["user_idx"] > 1000

        # Fetch cold-start recommendations
        res_cold_feed = client.get(f"/api/recommendations/{new_user['user_idx']}?k=10")
        assert res_cold_feed.status_code == 200
        cold_data = res_cold_feed.json()
        assert len(cold_data["rails"]["Top Picks For You"]) > 0

        # Check genre relevance of cold-start items
        cold_genres = [
            genre
            for item in cold_data["rails"]["Top Picks For You"]
            for genre in item["genres"]
        ]
        assert any(g in ["Film-Noir", "Mystery", "Crime", "Drama"] for g in cold_genres), \
            "Cold start recommendations should match onboarding preferences"

        # Step 7: Catalog Search and Movie Details
        res_search = client.get("/api/search?q=godfather")
        assert res_search.status_code == 200
        search_results = res_search.json()["results"]
        assert len(search_results) > 0
        gf_idx = search_results[0]["movie_idx"]

        res_detail = client.get(f"/api/movies/{gf_idx}")
        assert res_detail.status_code == 200
        detail_data = res_detail.json()
        assert detail_data["movie_idx"] == gf_idx
        assert len(detail_data["similar_movies"]) > 0

        # Step 8: User Insights & Observability
        res_insights = client.get(f"/api/users/{elena['user_idx']}/insights")
        assert res_insights.status_code == 200
        insights_data = res_insights.json()
        assert "retrieval_sources_breakdown" in insights_data
        assert "timing_breakdown_ms" in insights_data

        # Step 9: Offline Empirical Benchmark Matrix
        res_metrics = client.get("/api/metrics")
        assert res_metrics.status_code == 200
        metrics_data = res_metrics.json()
        assert "Two-Tower + LightGBM Ranker" in metrics_data["models"]
        tt_lgbm = metrics_data["models"]["Two-Tower + LightGBM Ranker"]
        assert tt_lgbm["ndcg@10"] > 0.02
        assert tt_lgbm["catalog_coverage@100"] > 0.90
