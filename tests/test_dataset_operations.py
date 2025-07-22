import unittest
from unittest.mock import Mock, patch
import requests
from plum_sdk import PlumClient, IOPair, IOPairMeta, Dataset


class TestDatasetOperations(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.base_url = "http://test.getplum.ai/v1"
        self.client = PlumClient(self.api_key, self.base_url)

    @patch("requests.get")
    def test_get_dataset_with_empty_data(self, mock_get):
        """Test getting a dataset with no pairs"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "empty_dataset",
            "data": [],
            "system_prompt": "Empty dataset prompt",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        result = self.client.get_dataset("empty_dataset")

        mock_get.assert_called_once_with(
            f"{self.base_url}/data/seed/empty_dataset", headers=self.client.headers
        )
        self.assertEqual(result.id, "empty_dataset")
        self.assertEqual(len(result.data), 0)
        self.assertEqual(result.system_prompt, "Empty dataset prompt")
        self.assertEqual(result.created_at, "2023-01-01T00:00:00Z")

    @patch("requests.get")
    def test_get_dataset_with_complex_metadata(self, mock_get):
        """Test getting a dataset with complex metadata and media"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "complex_dataset",
            "data": [
                {
                    "id": "pair_with_media",
                    "input": "Analyze this image",
                    "output": "This is an image analysis",
                    "metadata": {
                        "created_at": "2023-01-01T00:00:00Z",
                        "labels": ["image", "analysis", "test"],
                    },
                    "input_media": b"fake_image_bytes",
                    "use_media_mime_type": "image/jpeg",
                    "human_critique": "Good analysis",
                    "target_metric": "accuracy",
                },
                {
                    "id": "pair_minimal",
                    "input": "Simple text",
                    "output": "Simple response",
                    "metadata": {"created_at": "2023-01-02T00:00:00Z", "labels": []},
                },
            ],
            "system_prompt": "Complex system prompt",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        result = self.client.get_dataset("complex_dataset")

        # Verify dataset structure
        self.assertEqual(result.id, "complex_dataset")
        self.assertEqual(len(result.data), 2)

        # Verify first pair with full metadata
        pair1 = result.data[0]
        self.assertEqual(pair1.id, "pair_with_media")
        self.assertEqual(pair1.input, "Analyze this image")
        self.assertEqual(pair1.output, "This is an image analysis")
        self.assertEqual(pair1.input_media, b"fake_image_bytes")
        self.assertEqual(pair1.use_media_mime_type, "image/jpeg")
        self.assertEqual(pair1.human_critique, "Good analysis")
        self.assertEqual(pair1.target_metric, "accuracy")
        self.assertIsNotNone(pair1.metadata)
        self.assertEqual(pair1.metadata.created_at, "2023-01-01T00:00:00Z")
        self.assertEqual(pair1.metadata.labels, ["image", "analysis", "test"])

        # Verify second pair with minimal metadata
        pair2 = result.data[1]
        self.assertEqual(pair2.id, "pair_minimal")
        self.assertIsNone(pair2.input_media)
        self.assertIsNone(pair2.human_critique)
        self.assertEqual(pair2.metadata.labels, [])

    @patch("requests.get")
    def test_get_dataset_without_metadata(self, mock_get):
        """Test getting a dataset where pairs have no metadata"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "no_metadata_dataset",
            "data": [
                {
                    "id": "pair_no_meta",
                    "input": "Input without metadata",
                    "output": "Output without metadata",
                }
            ],
            "system_prompt": "System prompt",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        result = self.client.get_dataset("no_metadata_dataset")

        self.assertEqual(len(result.data), 1)
        pair = result.data[0]
        self.assertEqual(pair.id, "pair_no_meta")
        self.assertIsNone(pair.metadata)

    @patch("requests.get")
    def test_get_dataset_synthetic_data(self, mock_get):
        """Test getting synthetic dataset"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "synthetic_dataset",
            "data": [
                {
                    "id": "synthetic_pair",
                    "input": "Generated input",
                    "output": "Generated output",
                    "metadata": {
                        "created_at": "2023-01-01T00:00:00Z",
                        "labels": ["synthetic"],
                    },
                }
            ],
            "system_prompt": "Synthetic system prompt",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        result = self.client.get_dataset("synthetic_dataset", is_synthetic=True)

        mock_get.assert_called_once_with(
            f"{self.base_url}/data/synthetic/synthetic_dataset",
            headers=self.client.headers,
        )
        self.assertEqual(result.id, "synthetic_dataset")
        self.assertEqual(len(result.data), 1)

    @patch("requests.get")
    def test_get_dataset_http_error(self, mock_get):
        """Test handling HTTP errors when getting dataset"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "Not Found"
        )
        mock_get.return_value = mock_response

        with self.assertRaises(requests.exceptions.HTTPError):
            self.client.get_dataset("nonexistent_dataset")

    @patch("requests.get")
    def test_get_dataset_server_error(self, mock_get):
        """Test handling server errors when getting dataset"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "Internal Server Error"
        )
        mock_get.return_value = mock_response

        with self.assertRaises(requests.exceptions.HTTPError):
            self.client.get_dataset("dataset_id")

    @patch("requests.get")
    def test_get_pair_from_large_dataset(self, mock_get):
        """Test getting a specific pair from a large dataset"""
        # Create a large dataset with many pairs
        pairs_data = []
        for i in range(100):
            pairs_data.append(
                {
                    "id": f"pair_{i}",
                    "input": f"Input {i}",
                    "output": f"Output {i}",
                    "metadata": {
                        "created_at": f"2023-01-{i+1:02d}T00:00:00Z",
                        "labels": [f"label_{i}"],
                    },
                }
            )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "large_dataset",
            "data": pairs_data,
            "system_prompt": "Large dataset prompt",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        result = self.client.get_pair("large_dataset", "pair_50")

        self.assertEqual(result.id, "pair_50")
        self.assertEqual(result.input, "Input 50")
        self.assertEqual(result.output, "Output 50")

    @patch("requests.get")
    def test_get_pair_synthetic_data(self, mock_get):
        """Test getting a pair from synthetic dataset"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "synthetic_dataset",
            "data": [
                {
                    "id": "synthetic_pair",
                    "input": "Synthetic input",
                    "output": "Synthetic output",
                    "metadata": {
                        "created_at": "2023-01-01T00:00:00Z",
                        "labels": ["synthetic"],
                    },
                }
            ],
            "system_prompt": "Synthetic system prompt",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        result = self.client.get_pair(
            "synthetic_dataset", "synthetic_pair", is_synthetic=True
        )

        mock_get.assert_called_once_with(
            f"{self.base_url}/data/synthetic/synthetic_dataset",
            headers=self.client.headers,
        )
        self.assertEqual(result.id, "synthetic_pair")

    @patch("requests.get")
    def test_get_pair_case_sensitive(self, mock_get):
        """Test that pair ID matching is case sensitive"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "case_dataset",
            "data": [
                {
                    "id": "Pair_1",
                    "input": "Input 1",
                    "output": "Output 1",
                    "metadata": {"created_at": "2023-01-01T00:00:00Z", "labels": []},
                },
                {
                    "id": "pair_1",
                    "input": "Input 2",
                    "output": "Output 2",
                    "metadata": {"created_at": "2023-01-01T00:00:00Z", "labels": []},
                },
            ],
            "system_prompt": "Case sensitive dataset",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        # Should find the exact case match
        result = self.client.get_pair("case_dataset", "Pair_1")
        self.assertEqual(result.input, "Input 1")

        result = self.client.get_pair("case_dataset", "pair_1")
        self.assertEqual(result.input, "Input 2")

    @patch("requests.get")
    def test_get_pair_empty_dataset(self, mock_get):
        """Test getting a pair from an empty dataset"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "empty_dataset",
            "data": [],
            "system_prompt": "Empty dataset",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        with self.assertRaises(ValueError) as context:
            self.client.get_pair("empty_dataset", "any_pair")

        self.assertIn("Pair with ID 'any_pair' not found", str(context.exception))
        self.assertIn("empty_dataset", str(context.exception))

    @patch("requests.get")
    def test_get_pair_multiple_calls_same_dataset(self, mock_get):
        """Test that multiple calls to get_pair make separate API calls"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test_dataset",
            "data": [
                {
                    "id": "pair_1",
                    "input": "Input 1",
                    "output": "Output 1",
                    "metadata": {"created_at": "2023-01-01T00:00:00Z", "labels": []},
                },
                {
                    "id": "pair_2",
                    "input": "Input 2",
                    "output": "Output 2",
                    "metadata": {"created_at": "2023-01-01T00:00:00Z", "labels": []},
                },
            ],
            "system_prompt": "Test dataset",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        # Get multiple pairs from the same dataset
        pair1 = self.client.get_pair("test_dataset", "pair_1")
        pair2 = self.client.get_pair("test_dataset", "pair_2")

        # Should make separate API calls (current implementation)
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(pair1.id, "pair_1")
        self.assertEqual(pair2.id, "pair_2")

    def test_dataset_dataclass_instantiation(self):
        """Test that Dataset dataclass can be instantiated directly"""
        metadata = IOPairMeta(created_at="2023-01-01T00:00:00Z", labels=["test"])
        pair = IOPair(
            id="test_pair", input="test input", output="test output", metadata=metadata
        )
        dataset = Dataset(
            id="test_dataset",
            data=[pair],
            system_prompt="test prompt",
            created_at="2023-01-01T00:00:00Z",
        )

        self.assertEqual(dataset.id, "test_dataset")
        self.assertEqual(len(dataset.data), 1)
        self.assertEqual(dataset.data[0].id, "test_pair")
        self.assertEqual(dataset.system_prompt, "test prompt")

    def test_iopair_dataclass_defaults(self):
        """Test IOPair dataclass with default values"""
        pair = IOPair(id="minimal_pair", input="minimal input", output="minimal output")

        self.assertEqual(pair.id, "minimal_pair")
        self.assertEqual(pair.input, "minimal input")
        self.assertEqual(pair.output, "minimal output")
        self.assertIsNone(pair.metadata)
        self.assertIsNone(pair.input_media)
        self.assertIsNone(pair.use_media_mime_type)
        self.assertIsNone(pair.human_critique)
        self.assertIsNone(pair.target_metric)

    def test_iopair_meta_dataclass_defaults(self):
        """Test IOPairMeta dataclass with default values"""
        metadata = IOPairMeta()

        self.assertIsNone(metadata.created_at)
        self.assertIsNone(metadata.labels)

        # Test with partial values
        metadata_partial = IOPairMeta(created_at="2023-01-01T00:00:00Z")
        self.assertEqual(metadata_partial.created_at, "2023-01-01T00:00:00Z")
        self.assertIsNone(metadata_partial.labels)


if __name__ == "__main__":
    unittest.main()
