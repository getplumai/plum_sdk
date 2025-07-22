import unittest
from unittest.mock import Mock, patch
import requests
from plum_sdk import PlumClient, IOPair, IOPairMeta, Dataset, TrainingExample


class TestDatasetIntegration(unittest.TestCase):
    """Integration tests for dataset operations with real-world scenarios"""

    def setUp(self):
        self.api_key = "test_api_key"
        self.base_url = "http://test.getplum.ai/v1"
        self.client = PlumClient(self.api_key, self.base_url)

    @patch("requests.post")
    @patch("requests.get")
    def test_upload_then_retrieve_dataset(self, mock_get, mock_post):
        """Test uploading data and then retrieving it"""
        # Mock upload response
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {"id": "uploaded_dataset"}
        mock_post.return_value = mock_post_response

        # Mock get response
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "id": "uploaded_dataset",
            "data": [
                {
                    "id": "example_1",
                    "input": "What is AI?",
                    "output": "AI is artificial intelligence",
                    "metadata": {"created_at": "2023-01-01T00:00:00Z", "labels": []},
                },
                {
                    "id": "example_2",
                    "input": "How does ML work?",
                    "output": "ML uses algorithms to learn patterns",
                    "metadata": {"created_at": "2023-01-01T00:00:00Z", "labels": []},
                },
            ],
            "system_prompt": "You are a helpful AI assistant",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_get_response

        # Upload training examples
        examples = [
            TrainingExample(
                input="What is AI?", output="AI is artificial intelligence"
            ),
            TrainingExample(
                input="How does ML work?", output="ML uses algorithms to learn patterns"
            ),
        ]
        upload_result = self.client.upload_data(
            examples, "You are a helpful AI assistant"
        )

        # Retrieve the dataset
        dataset = self.client.get_dataset(upload_result.id)

        # Verify the round trip
        self.assertEqual(dataset.id, "uploaded_dataset")
        self.assertEqual(len(dataset.data), 2)
        self.assertEqual(dataset.system_prompt, "You are a helpful AI assistant")
        self.assertEqual(dataset.data[0].input, "What is AI?")
        self.assertEqual(dataset.data[1].input, "How does ML work?")

    @patch("requests.post")
    @patch("requests.get")
    def test_upload_pair_then_retrieve_dataset(self, mock_get, mock_post):
        """Test uploading a single pair and then retrieving the dataset"""
        # Mock upload pair response
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            "dataset_id": "existing_dataset",
            "pair_id": "new_pair",
        }
        mock_post.return_value = mock_post_response

        # Mock get dataset response
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "id": "existing_dataset",
            "data": [
                {
                    "id": "original_pair",
                    "input": "Original input",
                    "output": "Original output",
                    "metadata": {"created_at": "2023-01-01T00:00:00Z", "labels": []},
                },
                {
                    "id": "new_pair",
                    "input": "New input",
                    "output": "New output",
                    "metadata": {
                        "created_at": "2023-01-01T01:00:00Z",
                        "labels": ["new", "uploaded"],
                    },
                },
            ],
            "system_prompt": "You are a helpful assistant",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_get_response

        # Upload a single pair
        pair_result = self.client.upload_pair(
            "existing_dataset", "New input", "New output", labels=["new", "uploaded"]
        )

        # Retrieve the updated dataset
        dataset = self.client.get_dataset(pair_result.dataset_id)

        # Verify the dataset contains both pairs
        self.assertEqual(len(dataset.data), 2)
        self.assertEqual(dataset.data[1].id, "new_pair")
        self.assertEqual(dataset.data[1].metadata.labels, ["new", "uploaded"])

    @patch("requests.get")
    def test_retrieve_and_filter_pairs(self, mock_get):
        """Test retrieving a dataset and filtering pairs by various criteria"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "filter_dataset",
            "data": [
                {
                    "id": "pair_1",
                    "input": "Question 1",
                    "output": "Answer 1",
                    "metadata": {
                        "created_at": "2023-01-01T00:00:00Z",
                        "labels": ["category_A", "important"],
                    },
                },
                {
                    "id": "pair_2",
                    "input": "Question 2",
                    "output": "Answer 2",
                    "metadata": {
                        "created_at": "2023-01-02T00:00:00Z",
                        "labels": ["category_B"],
                    },
                },
                {
                    "id": "pair_3",
                    "input": "Question 3",
                    "output": "Answer 3",
                    "metadata": {
                        "created_at": "2023-01-03T00:00:00Z",
                        "labels": ["category_A", "complex"],
                    },
                },
            ],
            "system_prompt": "You are a helpful assistant",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        dataset = self.client.get_dataset("filter_dataset")

        # Filter pairs by label
        category_a_pairs = [
            pair
            for pair in dataset.data
            if pair.metadata and "category_A" in pair.metadata.labels
        ]
        self.assertEqual(len(category_a_pairs), 2)
        self.assertEqual(category_a_pairs[0].id, "pair_1")
        self.assertEqual(category_a_pairs[1].id, "pair_3")

        # Filter pairs by creation date
        recent_pairs = [
            pair
            for pair in dataset.data
            if pair.metadata and pair.metadata.created_at >= "2023-01-02T00:00:00Z"
        ]
        self.assertEqual(len(recent_pairs), 2)

        # Filter pairs with specific label
        important_pairs = [
            pair
            for pair in dataset.data
            if pair.metadata and "important" in pair.metadata.labels
        ]
        self.assertEqual(len(important_pairs), 1)
        self.assertEqual(important_pairs[0].id, "pair_1")

    @patch("requests.get")
    def test_get_multiple_pairs_from_dataset(self, mock_get):
        """Test retrieving multiple specific pairs from a dataset"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "multi_pair_dataset",
            "data": [
                {
                    "id": "pair_alpha",
                    "input": "Alpha input",
                    "output": "Alpha output",
                    "metadata": {
                        "created_at": "2023-01-01T00:00:00Z",
                        "labels": ["alpha"],
                    },
                },
                {
                    "id": "pair_beta",
                    "input": "Beta input",
                    "output": "Beta output",
                    "metadata": {
                        "created_at": "2023-01-01T00:00:00Z",
                        "labels": ["beta"],
                    },
                },
                {
                    "id": "pair_gamma",
                    "input": "Gamma input",
                    "output": "Gamma output",
                    "metadata": {
                        "created_at": "2023-01-01T00:00:00Z",
                        "labels": ["gamma"],
                    },
                },
            ],
            "system_prompt": "You are a helpful assistant",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        # Get multiple pairs
        pair_ids = ["pair_alpha", "pair_gamma"]
        pairs = []
        for pair_id in pair_ids:
            pair = self.client.get_pair("multi_pair_dataset", pair_id)
            pairs.append(pair)

        # Verify we got the right pairs
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].id, "pair_alpha")
        self.assertEqual(pairs[1].id, "pair_gamma")
        self.assertEqual(pairs[0].metadata.labels, ["alpha"])
        self.assertEqual(pairs[1].metadata.labels, ["gamma"])

    @patch("requests.get")
    def test_dataset_with_media_pairs(self, mock_get):
        """Test retrieving dataset with media-containing pairs"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "media_dataset",
            "data": [
                {
                    "id": "text_pair",
                    "input": "Text only input",
                    "output": "Text only output",
                    "metadata": {
                        "created_at": "2023-01-01T00:00:00Z",
                        "labels": ["text"],
                    },
                },
                {
                    "id": "image_pair",
                    "input": "Describe this image",
                    "output": "This is a beautiful landscape",
                    "metadata": {
                        "created_at": "2023-01-01T00:00:00Z",
                        "labels": ["image", "vision"],
                    },
                    "input_media": b"fake_image_data",
                    "use_media_mime_type": "image/jpeg",
                },
                {
                    "id": "audio_pair",
                    "input": "Transcribe this audio",
                    "output": "Hello world",
                    "metadata": {
                        "created_at": "2023-01-01T00:00:00Z",
                        "labels": ["audio", "transcription"],
                    },
                    "input_media": b"fake_audio_data",
                    "use_media_mime_type": "audio/wav",
                },
            ],
            "system_prompt": "You are a multimodal AI assistant",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        dataset = self.client.get_dataset("media_dataset")

        # Verify different media types
        self.assertEqual(len(dataset.data), 3)

        # Text only pair
        text_pair = dataset.data[0]
        self.assertIsNone(text_pair.input_media)
        self.assertIsNone(text_pair.use_media_mime_type)

        # Image pair
        image_pair = dataset.data[1]
        self.assertEqual(image_pair.input_media, b"fake_image_data")
        self.assertEqual(image_pair.use_media_mime_type, "image/jpeg")

        # Audio pair
        audio_pair = dataset.data[2]
        self.assertEqual(audio_pair.input_media, b"fake_audio_data")
        self.assertEqual(audio_pair.use_media_mime_type, "audio/wav")

    @patch("requests.get")
    def test_error_handling_in_dataset_operations(self, mock_get):
        """Test proper error handling in dataset operations"""
        # Test network error
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

        with self.assertRaises(requests.exceptions.ConnectionError):
            self.client.get_dataset("test_dataset")

        # Test timeout error
        mock_get.side_effect = requests.exceptions.Timeout("Request timeout")

        with self.assertRaises(requests.exceptions.Timeout):
            self.client.get_dataset("test_dataset")

        # Test malformed JSON response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.side_effect = None
        mock_get.return_value = mock_response

        with self.assertRaises(ValueError):
            self.client.get_dataset("test_dataset")

    def test_dataclass_serialization_compatibility(self):
        """Test that our dataclasses can be serialized and deserialized"""
        import json
        from dataclasses import asdict

        # Create a complex dataset
        metadata = IOPairMeta(
            created_at="2023-01-01T00:00:00Z", labels=["test", "serialization"]
        )
        pair = IOPair(
            id="test_pair",
            input="test input",
            output="test output",
            metadata=metadata,
            human_critique="Good response",
            target_metric="accuracy",
        )
        dataset = Dataset(
            id="test_dataset",
            data=[pair],
            system_prompt="Test system prompt",
            created_at="2023-01-01T00:00:00Z",
        )

        # Convert to dict and serialize
        dataset_dict = asdict(dataset)
        json_str = json.dumps(dataset_dict, default=str)

        # Deserialize
        deserialized_dict = json.loads(json_str)

        # Verify structure is preserved
        self.assertEqual(deserialized_dict["id"], "test_dataset")
        self.assertEqual(len(deserialized_dict["data"]), 1)
        self.assertEqual(deserialized_dict["data"][0]["id"], "test_pair")
        self.assertEqual(
            deserialized_dict["data"][0]["metadata"]["labels"],
            ["test", "serialization"],
        )


if __name__ == "__main__":
    unittest.main()
