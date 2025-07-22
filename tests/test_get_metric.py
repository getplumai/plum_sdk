import unittest
from unittest.mock import Mock, patch
import json
from plum_sdk import PlumClient, DetailedMetricsResponse, MetricDefinition
import requests


class TestGetMetricFunction(unittest.TestCase):
    """Test the get_metric function in PlumClient"""

    def setUp(self):
        """Set up test fixtures"""
        self.client = PlumClient(api_key="test_key", base_url="https://test.api.com/v1")

    def test_get_metric_function_exists(self):
        """Test that the get_metric function exists and is callable"""
        self.assertTrue(hasattr(self.client, "get_metric"))
        self.assertTrue(callable(getattr(self.client, "get_metric")))

    @patch("requests.get")
    def test_get_metric_success_single_definition(self, mock_get):
        """Test successful get_metric call with single definition"""
        # Mock successful response with single definition
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metrics_id": "test_metric_id",
            "definitions": [
                {
                    "id": "accuracy",
                    "name": "Accuracy",
                    "description": "Measures response accuracy",
                }
            ],
            "system_prompt": "You are a helpful assistant",
            "num_metrics": 1,
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        result = self.client.get_metric("test_metric_id")

        # Verify the request was made correctly
        mock_get.assert_called_once_with(
            "https://test.api.com/v1/question/test_metric_id",
            headers=self.client.headers,
        )

        # Verify the response structure
        self.assertIsInstance(result, DetailedMetricsResponse)
        self.assertEqual(result.metrics_id, "test_metric_id")
        self.assertEqual(len(result.definitions), 1)
        self.assertEqual(result.system_prompt, "You are a helpful assistant")
        self.assertEqual(result.metric_count, 1)
        self.assertEqual(result.created_at, "2023-01-01T00:00:00Z")

        # Verify the definition details
        definition = result.definitions[0]
        self.assertIsInstance(definition, MetricDefinition)
        self.assertEqual(definition.id, "accuracy")
        self.assertEqual(definition.name, "Accuracy")
        self.assertEqual(definition.description, "Measures response accuracy")

    @patch("requests.get")
    def test_get_metric_success_multiple_definitions(self, mock_get):
        """Test successful get_metric call with multiple definitions"""
        # Mock successful response with multiple definitions
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metrics_id": "test_metric_id",
            "definitions": [
                {
                    "id": "accuracy",
                    "name": "Accuracy",
                    "description": "Measures response accuracy",
                },
                {
                    "id": "relevance",
                    "name": "Relevance",
                    "description": "Measures response relevance",
                },
            ],
            "system_prompt": "You are a helpful assistant",
            "num_metrics": 2,
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        result = self.client.get_metric("test_metric_id")

        # Verify the response structure
        self.assertIsInstance(result, DetailedMetricsResponse)
        self.assertEqual(result.metrics_id, "test_metric_id")
        self.assertEqual(len(result.definitions), 2)
        self.assertEqual(result.metric_count, 2)

        # Verify the first definition
        definition1 = result.definitions[0]
        self.assertIsInstance(definition1, MetricDefinition)
        self.assertEqual(definition1.id, "accuracy")
        self.assertEqual(definition1.name, "Accuracy")
        self.assertEqual(definition1.description, "Measures response accuracy")

        # Verify the second definition
        definition2 = result.definitions[1]
        self.assertIsInstance(definition2, MetricDefinition)
        self.assertEqual(definition2.id, "relevance")
        self.assertEqual(definition2.name, "Relevance")
        self.assertEqual(definition2.description, "Measures response relevance")

    @patch("requests.get")
    def test_get_metric_success_string_definitions(self, mock_get):
        """Test successful get_metric call with string definitions (legacy format)"""
        # Mock successful response with string definitions
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metrics_id": "test_metric_id",
            "definitions": [
                "Measures response accuracy",
                "Measures response relevance",
            ],
            "system_prompt": "You are a helpful assistant",
            "num_metrics": 2,
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        result = self.client.get_metric("test_metric_id")

        # Verify the response structure
        self.assertIsInstance(result, DetailedMetricsResponse)
        self.assertEqual(result.metrics_id, "test_metric_id")
        self.assertEqual(len(result.definitions), 2)
        self.assertEqual(result.metric_count, 2)

        # Verify the first definition (auto-generated from string)
        definition1 = result.definitions[0]
        self.assertIsInstance(definition1, MetricDefinition)
        self.assertEqual(definition1.id, "metric_0")
        self.assertEqual(definition1.name, "Metric 1")
        self.assertEqual(definition1.description, "Measures response accuracy")

        # Verify the second definition (auto-generated from string)
        definition2 = result.definitions[1]
        self.assertIsInstance(definition2, MetricDefinition)
        self.assertEqual(definition2.id, "metric_1")
        self.assertEqual(definition2.name, "Metric 2")
        self.assertEqual(definition2.description, "Measures response relevance")

    @patch("requests.get")
    def test_get_metric_success_minimal_response(self, mock_get):
        """Test successful get_metric call with minimal response data"""
        # Mock successful response with minimal data
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metrics_id": "test_metric_id",
            "definitions": ["Basic metric definition"],
        }
        mock_get.return_value = mock_response

        result = self.client.get_metric("test_metric_id")

        # Verify the response structure with defaults
        self.assertIsInstance(result, DetailedMetricsResponse)
        self.assertEqual(result.metrics_id, "test_metric_id")
        self.assertEqual(len(result.definitions), 1)
        self.assertEqual(result.system_prompt, None)
        self.assertEqual(
            result.metric_count, 1
        )  # Should default to length of definitions
        self.assertEqual(result.created_at, None)

    @patch("requests.get")
    def test_get_metric_success_missing_metrics_id(self, mock_get):
        """Test successful get_metric call when metrics_id is missing in response"""
        # Mock successful response without metrics_id
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "definitions": [
                {
                    "id": "accuracy",
                    "name": "Accuracy",
                    "description": "Measures response accuracy",
                }
            ],
            "system_prompt": "You are a helpful assistant",
        }
        mock_get.return_value = mock_response

        result = self.client.get_metric("test_metric_id")

        # Verify the response uses the passed metrics_id as fallback
        self.assertIsInstance(result, DetailedMetricsResponse)
        self.assertEqual(result.metrics_id, "test_metric_id")

    @patch("requests.get")
    def test_get_metric_http_error_404(self, mock_get):
        """Test get_metric raises HTTPError on 404 response"""
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Not Found"
        )
        mock_get.return_value = mock_response

        with self.assertRaises(requests.exceptions.HTTPError):
            self.client.get_metric("nonexistent_metric_id")

    @patch("requests.get")
    def test_get_metric_http_error_500(self, mock_get):
        """Test get_metric raises HTTPError on 500 response"""
        # Mock 500 response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Internal Server Error"
        )
        mock_get.return_value = mock_response

        with self.assertRaises(requests.exceptions.HTTPError):
            self.client.get_metric("test_metric_id")

    @patch("requests.get")
    def test_get_metric_network_error(self, mock_get):
        """Test get_metric raises RequestException on network error"""
        # Mock network error
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        with self.assertRaises(requests.exceptions.RequestException):
            self.client.get_metric("test_metric_id")

    @patch("requests.get")
    def test_get_metric_json_decode_error(self, mock_get):
        """Test get_metric handles JSON decode error gracefully"""
        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response

        with self.assertRaises(json.JSONDecodeError):
            self.client.get_metric("test_metric_id")

    @patch("requests.get")
    def test_get_metric_empty_definitions_list(self, mock_get):
        """Test get_metric with empty definitions list"""
        # Mock successful response with empty definitions
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metrics_id": "test_metric_id",
            "definitions": [],
            "system_prompt": "You are a helpful assistant",
            "num_metrics": 0,
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        result = self.client.get_metric("test_metric_id")

        # Verify the response structure
        self.assertIsInstance(result, DetailedMetricsResponse)
        self.assertEqual(result.metrics_id, "test_metric_id")
        self.assertEqual(len(result.definitions), 0)
        self.assertEqual(result.metric_count, 0)

    def test_get_metric_empty_metrics_id(self):
        """Test get_metric with empty metrics_id parameter"""
        # This should make a request to /question/ which may not be valid
        # but we test the behavior anyway
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "400 Bad Request"
            )
            mock_get.return_value = mock_response

            with self.assertRaises(requests.exceptions.HTTPError):
                self.client.get_metric("")


if __name__ == "__main__":
    unittest.main()
