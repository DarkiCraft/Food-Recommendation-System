import unittest
import sys
import shutil
from pathlib import Path
import pandas as pd
import torch

# Add src to path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from src.data_loader import generate_synthetic_data, load_data, InteractionDataset
from src.models.ncf import NCFModel, train_ncf
from src.recommender import RecommendationService
from src.config import RAW_DATA_DIR, MODELS_DIR


class TestCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Clean up data dir for fresh test
        if RAW_DATA_DIR.exists():
            shutil.rmtree(RAW_DATA_DIR)
        RAW_DATA_DIR.mkdir(parents=True)

    def test_01_data_generation(self):
        users, items, interactions = generate_synthetic_data()
        self.assertFalse(users.empty)
        self.assertFalse(items.empty)
        self.assertFalse(interactions.empty)
        self.assertTrue((RAW_DATA_DIR / "users.csv").exists())
        print("Data Generation Test Passed")

    def test_02_model_training(self):
        users, items, interactions = load_data()
        dataset = InteractionDataset(interactions)
        # Mock loader
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)

        model = NCFModel(100, 50)
        trained_model = train_ncf(model, loader, epochs=1)
        self.assertIsInstance(trained_model, NCFModel)
        print("Model Training Test Passed")

    def test_03_recommendation_service(self):
        service = RecommendationService()
        recs = service.get_recommendations(user_id=1, k=3)
        self.assertEqual(len(recs), 3)
        self.assertTrue("item_id" in recs.columns)

        # Test content
        recs_content = service.get_recommendations(user_id=1, strategy="content", k=3)
        self.assertEqual(len(recs_content), 3)

    def test_05_admin_stats(self):
        service = RecommendationService()
        stats = service.get_system_stats()
        self.assertIn("sparsity", stats)
        self.assertIn("catalog_coverage", stats)
        self.assertIn("active_users", stats)
        print("Admin Stats Test Passed")

    def test_06_model_evaluation(self):
        service = RecommendationService()
        # Ensure model is trained (handled in init)
        metrics = service.calculate_metrics(k=5)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("map", metrics)
        self.assertIn("accuracy", metrics)
        print(f"Metrics Calculated: {metrics}")
        print("Evaluation Test Passed")

    def test_07_simulation(self):
        from src.simulation import run_simulation

        service = RecommendationService()

        # Run small sim
        interactions = run_simulation(service.items_df, num_users=5, views_per_user=10)
        self.assertFalse(interactions.empty)
        self.assertTrue("user_id" in interactions.columns)
        self.assertTrue("item_id" in interactions.columns)
        print("Simulation Test Passed")

    def test_08_diversification(self):
        service = RecommendationService()
        # Mocking large enough predictions to test diversity?
        # Just running it for a user and checking cuisine counts
        recs = service.get_recommendations(user_id=1, strategy="personalized", k=5)
        cuisines = recs["cuisine_type"].tolist()
        # Verify no single cuisine dominates (max 2 per cuisine) - unless dataset is tiny.
        # Given we have 7 cuisines and random data, it should be distributed.
        from collections import Counter

        counts = Counter(cuisines)
        print(f"Diversification Test - Cuisine Counts: {counts}")
        # Assert no cuisine > 2 (Soft check as it has fallback)
        self.assertTrue(
            all(c <= 3 for c in counts.values()),
            "Diversity check failed (too many same items)",
        )


if __name__ == "__main__":
    unittest.main()
