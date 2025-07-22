import unittest
from plum_sdk.plum_sdk import PlumClient
from plum_sdk.models import UploadResponse, MetricScore, ScoringPair


class TestResponseFiltering(unittest.TestCase):
    """Test the response filtering functionality."""

    def setUp(self):
        """Set up test client."""
        self.client = PlumClient("dummy_key")

    def test_upload_response_filtering(self):
        """Test UploadResponse filtering."""
        mock_response = {
            "id": "test123",
            "extra_field": "should_be_filtered_out",
            "another_field": {"nested": "data"},
            "random_array": [1, 2, 3]
        }
        
        filtered = self.client._filter_response_for_dataclass(mock_response, UploadResponse)
        
        # Should only contain 'id' field
        self.assertEqual(filtered, {"id": "test123"})
        self.assertNotIn("extra_field", filtered)
        self.assertNotIn("another_field", filtered)
        self.assertNotIn("random_array", filtered)

    def test_metric_score_filtering(self):
        """Test MetricScore filtering."""
        metric_response = {
            "metric": "accuracy",
            "mean_score": 0.85,
            "std_dev": 0.1,
            "ci_low": 0.75,
            "ci_high": 0.95,
            "ci_confidence": 0.95,
            "median_score": 0.87,
            "min_score": 0.6,
            "max_score": 1.0,
            "lowest_scoring_pairs": [],
            "extra_field": "should_be_filtered",
            "unwanted_data": {"complex": "object"}
        }
        
        filtered_metric = self.client._filter_response_for_dataclass(metric_response, MetricScore)
        
        # Should not contain extra fields
        self.assertNotIn("extra_field", filtered_metric)
        self.assertNotIn("unwanted_data", filtered_metric)
        
        # Should contain valid fields
        self.assertIn("metric", filtered_metric)
        self.assertIn("mean_score", filtered_metric)
        self.assertIn("std_dev", filtered_metric)
        self.assertIn("ci_low", filtered_metric)
        self.assertIn("ci_high", filtered_metric)
        self.assertIn("ci_confidence", filtered_metric)
        self.assertIn("median_score", filtered_metric)
        self.assertIn("min_score", filtered_metric)
        self.assertIn("max_score", filtered_metric)
        self.assertIn("lowest_scoring_pairs", filtered_metric)

    def test_scoring_pair_filtering(self):
        """Test ScoringPair filtering."""
        pair_response = {
            "pair_id": "pair123",
            "score_reason": "Low accuracy",
            "irrelevant_field": "should_be_removed",
            "metadata": {"timestamp": "2023-01-01"}
        }
        
        filtered_pair = self.client._filter_response_for_dataclass(pair_response, ScoringPair)
        
        # Should only contain pair_id and score_reason
        self.assertEqual(filtered_pair, {"pair_id": "pair123", "score_reason": "Low accuracy"})
        self.assertNotIn("irrelevant_field", filtered_pair)
        self.assertNotIn("metadata", filtered_pair)

    def test_filtering_preserves_valid_fields(self):
        """Test that filtering preserves all valid fields and removes only invalid ones."""
        # Test with a mix of valid and invalid fields
        mixed_response = {
            "id": "valid_id",
            "invalid_field_1": "should_be_removed",
            "invalid_field_2": {"nested": "data"}
        }
        
        filtered = self.client._filter_response_for_dataclass(mixed_response, UploadResponse)
        
        # Should preserve valid field
        self.assertIn("id", filtered)
        self.assertEqual(filtered["id"], "valid_id")
        
        # Should remove invalid fields
        self.assertNotIn("invalid_field_1", filtered)
        self.assertNotIn("invalid_field_2", filtered)

    def test_filtering_with_empty_response(self):
        """Test filtering behavior with empty response."""
        empty_response = {}
        
        filtered = self.client._filter_response_for_dataclass(empty_response, UploadResponse)
        
        self.assertEqual(filtered, {})

    def test_filtering_with_all_valid_fields(self):
        """Test filtering when all fields are valid."""
        valid_response = {"id": "test123"}
        
        filtered = self.client._filter_response_for_dataclass(valid_response, UploadResponse)
        
        self.assertEqual(filtered, valid_response)


if __name__ == "__main__":
    unittest.main()
