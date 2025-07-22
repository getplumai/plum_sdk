import unittest
from unittest.mock import Mock, patch
import json
from plum_sdk import (
    PlumClient,
    MetricsListResponse,
    DetailedMetricsResponse,
    MetricDefinition,
)
import requests


class TestListMetricsFunction(unittest.TestCase):
    """Test the list_metrics function in PlumClient"""

    def setUp(self):
        """Set up test fixtures"""
        self.client = PlumClient(api_key="test_key", base_url="https://test.api.com/v1")

    def test_list_metrics_function_exists(self):
        """Test that the list_metrics function exists and is callable"""
        self.assertTrue(hasattr(self.client, "list_metrics"))
        self.assertTrue(callable(getattr(self.client, "list_metrics")))

    @patch("requests.get")
    def test_list_metrics_success_empty_response(self, mock_get):
        """Test successful list_metrics call with empty response"""
        # Mock successful response with empty metrics
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"metrics": {}, "total_count": 0}
        mock_get.return_value = mock_response

        result = self.client.list_metrics()

        # Verify the request was made correctly
        mock_get.assert_called_once_with(
            "https://test.api.com/v1/list_questions", headers=self.client.headers
        )

        # Verify the response structure
        self.assertIsInstance(result, MetricsListResponse)
        self.assertEqual(result.total_count, 0)
        self.assertEqual(len(result.metrics), 0)

    @patch("requests.get")
    def test_list_metrics_success_single_metric(self, mock_get):
        """Test successful list_metrics call with single metric"""
        # Mock successful response with single metric
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metrics": {
                "test_metric_id": {
                    "metrics_id": "test_metric_id",
                    "definitions": [
                        {
                            "id": "accuracy",
                            "name": "Accuracy",
                            "description": "Measures response accuracy",
                        }
                    ],
                    "system_prompt": "You are a helpful assistant",
                    "metric_count": 1,
                    "created_at": "2023-01-01T00:00:00Z",
                }
            },
            "total_count": 1,
        }
        mock_get.return_value = mock_response

        result = self.client.list_metrics()

        # Verify the response structure
        self.assertIsInstance(result, MetricsListResponse)
        self.assertEqual(result.total_count, 1)
        self.assertEqual(len(result.metrics), 1)

        # Verify the metric details
        metric = result.metrics["test_metric_id"]
        self.assertIsInstance(metric, DetailedMetricsResponse)
        self.assertEqual(metric.metrics_id, "test_metric_id")
        self.assertEqual(len(metric.definitions), 1)
        self.assertEqual(metric.system_prompt, "You are a helpful assistant")
        self.assertEqual(metric.metric_count, 1)
        self.assertEqual(metric.created_at, "2023-01-01T00:00:00Z")

        # Verify the definition details
        definition = metric.definitions[0]
        self.assertIsInstance(definition, MetricDefinition)
        self.assertEqual(definition.id, "accuracy")
        self.assertEqual(definition.name, "Accuracy")
        self.assertEqual(definition.description, "Measures response accuracy")

    @patch("requests.get")
    def test_list_metrics_success_multiple_metrics(self, mock_get):
        """Test successful list_metrics call with multiple metrics"""
        # Mock successful response with multiple metrics
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metrics": {
                "qa_metrics": {
                    "metrics_id": "qa_metrics",
                    "definitions": [
                        {
                            "id": "accuracy",
                            "name": "Accuracy",
                            "description": "Measures accuracy",
                        },
                        {
                            "id": "relevance",
                            "name": "Relevance",
                            "description": "Measures relevance",
                        },
                    ],
                    "system_prompt": "You are a Q&A assistant",
                    "metric_count": 2,
                    "created_at": "2023-01-01T00:00:00Z",
                },
                "writing_metrics": {
                    "metrics_id": "writing_metrics",
                    "definitions": [
                        {
                            "id": "clarity",
                            "name": "Clarity",
                            "description": "Measures clarity",
                        }
                    ],
                    "system_prompt": "You are a writing assistant",
                    "metric_count": 1,
                    "created_at": "2023-01-02T00:00:00Z",
                },
            },
            "total_count": 2,
        }
        mock_get.return_value = mock_response

        result = self.client.list_metrics()

        # Verify the response structure
        self.assertIsInstance(result, MetricsListResponse)
        self.assertEqual(result.total_count, 2)
        self.assertEqual(len(result.metrics), 2)

        # Verify QA metrics
        qa_metrics = result.metrics["qa_metrics"]
        self.assertEqual(qa_metrics.metrics_id, "qa_metrics")
        self.assertEqual(len(qa_metrics.definitions), 2)
        self.assertEqual(qa_metrics.metric_count, 2)

        # Verify writing metrics
        writing_metrics = result.metrics["writing_metrics"]
        self.assertEqual(writing_metrics.metrics_id, "writing_metrics")
        self.assertEqual(len(writing_metrics.definitions), 1)
        self.assertEqual(writing_metrics.metric_count, 1)

    @patch("requests.get")
    def test_list_metrics_string_definitions(self, mock_get):
        """Test list_metrics with string definitions instead of objects"""
        # Mock successful response with string definitions
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metrics": {
                "string_metrics": {
                    "metrics_id": "string_metrics",
                    "definitions": [
                        "This is a string definition for accuracy",
                        "This is another string definition for relevance",
                    ],
                    "metric_count": 2,
                    "created_at": "2023-01-01T00:00:00Z",
                }
            },
            "total_count": 1,
        }
        mock_get.return_value = mock_response

        result = self.client.list_metrics()

        # Verify the response structure
        self.assertIsInstance(result, MetricsListResponse)
        self.assertEqual(result.total_count, 1)

        # Verify string definitions are converted properly
        metric = result.metrics["string_metrics"]
        self.assertEqual(len(metric.definitions), 2)

        # Verify first definition
        def1 = metric.definitions[0]
        self.assertEqual(def1.id, "metric_0")
        self.assertEqual(def1.name, "Metric 1")
        self.assertEqual(def1.description, "This is a string definition for accuracy")

        # Verify second definition
        def2 = metric.definitions[1]
        self.assertEqual(def2.id, "metric_1")
        self.assertEqual(def2.name, "Metric 2")
        self.assertEqual(
            def2.description, "This is another string definition for relevance"
        )

    @patch("requests.get")
    def test_list_metrics_mixed_definitions(self, mock_get):
        """Test list_metrics with mixed definition formats"""
        # Mock successful response with mixed definition formats
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metrics": {
                "mixed_metrics": {
                    "metrics_id": "mixed_metrics",
                    "definitions": [
                        {
                            "id": "accuracy",
                            "name": "Accuracy",
                            "description": "Measures accuracy",
                        },
                        "This is a string definition",
                        {
                            "id": "relevance",
                            "name": "Relevance",
                            "text": "Uses text field instead of description",
                        },
                    ],
                    "metric_count": 3,
                }
            },
            "total_count": 1,
        }
        mock_get.return_value = mock_response

        result = self.client.list_metrics()

        # Verify the response structure
        metric = result.metrics["mixed_metrics"]
        self.assertEqual(len(metric.definitions), 3)

        # Verify object definition
        def1 = metric.definitions[0]
        self.assertEqual(def1.id, "accuracy")
        self.assertEqual(def1.name, "Accuracy")
        self.assertEqual(def1.description, "Measures accuracy")

        # Verify string definition
        def2 = metric.definitions[1]
        self.assertEqual(def2.id, "metric_1")
        self.assertEqual(def2.name, "Metric 2")
        self.assertEqual(def2.description, "This is a string definition")

        # Verify object definition with text field
        def3 = metric.definitions[2]
        self.assertEqual(def3.id, "relevance")
        self.assertEqual(def3.name, "Relevance")
        self.assertEqual(def3.description, "Uses text field instead of description")

    @patch("requests.get")
    def test_list_metrics_http_error(self, mock_get):
        """Test list_metrics with HTTP error response"""
        # Mock HTTP error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError("Server Error")
        mock_get.return_value = mock_response

        # Verify that HTTPError is raised
        with self.assertRaises(requests.HTTPError):
            self.client.list_metrics()

    @patch("requests.get")
    def test_list_metrics_unauthorized(self, mock_get):
        """Test list_metrics with unauthorized response"""
        # Mock unauthorized response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.HTTPError("Unauthorized")
        mock_get.return_value = mock_response

        # Verify that HTTPError is raised
        with self.assertRaises(requests.HTTPError):
            self.client.list_metrics()

    @patch("requests.get")
    def test_list_metrics_malformed_response(self, mock_get):
        """Test list_metrics with malformed JSON response"""
        # Mock response with malformed JSON
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response

        # Verify that JSONDecodeError is raised
        with self.assertRaises(json.JSONDecodeError):
            self.client.list_metrics()

    @patch("requests.get")
    def test_list_metrics_missing_fields(self, mock_get):
        """Test list_metrics with missing fields in response"""
        # Mock response with missing fields
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metrics": {
                "incomplete_metric": {
                    "metrics_id": "incomplete_metric",
                    # Missing definitions field
                    "metric_count": 1,
                }
            }
            # Missing total_count field
        }
        mock_get.return_value = mock_response

        result = self.client.list_metrics()

        # Verify the response handles missing fields gracefully
        self.assertIsInstance(result, MetricsListResponse)
        self.assertEqual(
            result.total_count, 1
        )  # Should default to length of metrics dict

        metric = result.metrics["incomplete_metric"]
        self.assertEqual(len(metric.definitions), 0)  # Should default to empty list

    @patch("requests.get")
    def test_list_metrics_correct_headers(self, mock_get):
        """Test that list_metrics uses correct headers"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"metrics": {}, "total_count": 0}
        mock_get.return_value = mock_response

        # Create client with custom headers
        client = PlumClient(api_key="custom_key", base_url="https://custom.api.com/v1")
        client.list_metrics()

        # Verify correct headers were used
        expected_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "custom_key",
        }
        mock_get.assert_called_once_with(
            "https://custom.api.com/v1/list_questions", headers=expected_headers
        )

    @patch("requests.get")
    def test_list_metrics_correct_url(self, mock_get):
        """Test that list_metrics uses correct URL"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"metrics": {}, "total_count": 0}
        mock_get.return_value = mock_response

        # Test with custom base URL
        client = PlumClient(
            api_key="test_key", base_url="https://custom.example.com/api/v2"
        )
        client.list_metrics()

        # Verify correct URL was used
        mock_get.assert_called_once_with(
            "https://custom.example.com/api/v2/list_questions", headers=client.headers
        )


if __name__ == "__main__":
    unittest.main()
