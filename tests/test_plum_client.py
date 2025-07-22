import unittest
from unittest.mock import Mock, patch
import requests
from plum_sdk import PlumClient, TrainingExample
from plum_sdk.models import (
    PairUploadResponse,
    TrainingExample,
    UploadResponse,
    MetricsQuestions,
    MetricsResponse,
    IOPair,
    IOPairMeta,
    Dataset,
)


class TestPlumClient(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.base_url = "http://test.getplum.ai/v1"
        self.client = PlumClient(self.api_key, self.base_url)

    def test_init(self):
        self.assertEqual(self.client.api_key, self.api_key)
        self.assertEqual(self.client.base_url, self.base_url)

    @patch("requests.post")
    def test_upload_data_success(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "data:0:0000000"}
        mock_post.return_value = mock_response

        examples = [
            TrainingExample(input="test input 1", output="test output 1"),
            TrainingExample(input="test input 2", output="test output 2"),
        ]
        system_prompt = "test system prompt"

        result = self.client.upload_data(examples, system_prompt)

        mock_post.assert_called_once()
        self.assertEqual(result, UploadResponse(id="data:0:0000000"))

    @patch("requests.post")
    def test_upload_data_failure(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError
        mock_post.return_value = mock_response

        examples = [TrainingExample(input="test", output="test")]
        system_prompt = "test"

        with self.assertRaises(requests.exceptions.HTTPError):
            self.client.upload_data(examples, system_prompt)

    @patch("requests.post")
    def test_generate_metric_questions(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "metrics_id": "eval:metrics:0:0000",
            "definitions": [
                "Is the code maintainable?",
                "Is the code well-documented?",
            ],
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = self.client.generate_metric_questions(
            "Generate maintainable code. It should be well-documented."
        )

        assert isinstance(result, MetricsQuestions)
        assert len(result.definitions) == 2
        assert "maintainable" in result.definitions[0]

        mock_post.assert_called_once()

    @patch("requests.post")
    def test_define_metric_questions(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {"metrics_id": "eval:metrics:0:000000"}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        questions = ["Is the code readable?", "Are there tests?"]
        result = self.client.define_metric_questions(questions)

        assert isinstance(result, MetricsResponse)
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_evaluate(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "eval_results_id": "eval:results:0:000000",
            "pair_count": 100,
            "scores": [
                {
                    "metric": "Is the code readable?",
                    "mean_score": 5,
                    "std_dev": 0,
                    "ci_low": 5,
                    "ci_high": 5,
                    "ci_confidence": 0.95,
                    "median_score": 5,
                    "min_score": 5,
                    "max_score": 5,
                    "lowest_scoring_pairs": [],
                }
            ],
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

    @patch("requests.post")
    def test_upload_pair(self, mock_post):
        # Setup
        test_api_key = "test-api-key"
        test_dataset_id = "test-dataset-id"
        test_input = "This is a test input"
        test_output = "This is a test output"
        test_pair_id = "test-pair-id"
        test_labels = ["label1", "label2"]

        expected_url = "https://beta.getplum.ai/v1/data/seed/test-dataset-id/pair"
        expected_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": test_api_key,
        }
        expected_payload = {
            "input": test_input,
            "output": test_output,
            "labels": test_labels,
            "id": test_pair_id,
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "dataset_id": test_dataset_id,
            "pair_id": test_pair_id,
        }

        mock_post.return_value = mock_response

        # Execute
        client = PlumClient(test_api_key)
        result = client.upload_pair(
            dataset_id=test_dataset_id,
            input_text=test_input,
            output_text=test_output,
            pair_id=test_pair_id,
            labels=test_labels,
        )

        # Verify
        mock_post.assert_called_once_with(
            expected_url, headers=expected_headers, json=expected_payload
        )
        assert isinstance(result, PairUploadResponse)
        assert result.dataset_id == test_dataset_id
        assert result.pair_id == test_pair_id

    @patch("requests.post")
    def test_upload_pair_without_optional_params(self, mock_post):
        # Setup
        test_api_key = "test-api-key"
        test_dataset_id = "test-dataset-id"
        test_input = "This is a test input"
        test_output = "This is a test output"

        expected_url = "https://beta.getplum.ai/v1/data/seed/test-dataset-id/pair"
        expected_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": test_api_key,
        }
        expected_payload = {"input": test_input, "output": test_output, "labels": []}

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "dataset_id": test_dataset_id,
            "pair_id": "auto-generated-id",
        }

        mock_post.return_value = mock_response

        # Execute
        client = PlumClient(test_api_key)
        result = client.upload_pair(
            dataset_id=test_dataset_id, input_text=test_input, output_text=test_output
        )

        # Verify
        mock_post.assert_called_once_with(
            expected_url, headers=expected_headers, json=expected_payload
        )
        assert isinstance(result, PairUploadResponse)
        assert result.dataset_id == test_dataset_id
        assert result.pair_id == "auto-generated-id"

    @patch("requests.post")
    def test_upload_pair_error_handling(self, mock_post):
        # Setup
        test_api_key = "test-api-key"
        test_dataset_id = "test-dataset-id"
        test_input = "This is a test input"
        test_output = "This is a test output"

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "Dataset not found"
        )

        mock_post.return_value = mock_response

        # Execute & Verify
        client = PlumClient(test_api_key)
        with self.assertRaises(requests.exceptions.HTTPError):
            client.upload_pair(
                dataset_id=test_dataset_id,
                input_text=test_input,
                output_text=test_output,
            )

    @patch("requests.post")
    def test_evaluate_with_pair_query_params(self, mock_post):
        # Setup
        mock_response = Mock()
        mock_response.json.return_value = {
            "eval_results_id": "eval:results:0:000000",
            "pair_count": 50,
            "scores": [
                {
                    "metric": "Is the code readable?",
                    "mean_score": 5,
                    "std_dev": 0,
                    "ci_low": 5,
                    "ci_high": 5,
                    "ci_confidence": 0.95,
                    "median_score": 5,
                    "min_score": 5,
                    "max_score": 5,
                    "lowest_scoring_pairs": [],
                }
            ],
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Test with pair query parameters
        result = self.client.evaluate(
            data_id="data:0:123456",
            metrics_id="eval:metrics:0:000000",
            latest_n_pairs=50,
            pair_labels=["geography"],
        )

        # Verify the request was made with correct payload
        expected_payload = {
            "seed_data_id": "data:0:123456",
            "metrics_id": "eval:metrics:0:000000",
            "pair_query": {"latest_n_pairs": 50, "pair_labels": ["geography"]},
        }

        mock_post.assert_called_once_with(
            f"{self.base_url}/evaluate",
            json=expected_payload,
            headers=self.client.headers,
        )

    @patch("requests.post")
    def test_evaluate_with_last_n_seconds(self, mock_post):
        # Setup
        mock_response = Mock()
        mock_response.json.return_value = {
            "eval_results_id": "eval:results:0:000000",
            "pair_count": 25,
            "scores": [
                {
                    "metric": "Is the code readable?",
                    "mean_score": 4.5,
                    "std_dev": 0.5,
                    "ci_low": 4.2,
                    "ci_high": 4.8,
                    "ci_confidence": 0.95,
                    "median_score": 4.5,
                    "min_score": 4,
                    "max_score": 5,
                    "lowest_scoring_pairs": [],
                }
            ],
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Test with last_n_seconds parameter
        result = self.client.evaluate(
            data_id="data:0:123456",
            metrics_id="eval:metrics:0:000000",
            last_n_seconds=3600,  # Last hour
        )

        # Verify the request was made with correct payload
        expected_payload = {
            "seed_data_id": "data:0:123456",
            "metrics_id": "eval:metrics:0:000000",
            "pair_query": {"last_n_seconds": 3600},
        }

        mock_post.assert_called_once_with(
            f"{self.base_url}/evaluate",
            json=expected_payload,
            headers=self.client.headers,
        )

    @patch("requests.post")
    def test_evaluate_with_all_pair_query_params(self, mock_post):
        # Setup
        mock_response = Mock()
        mock_response.json.return_value = {
            "eval_results_id": "eval:results:0:000000",
            "pair_count": 10,
            "scores": [],
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Test with all pair query parameters
        result = self.client.evaluate(
            data_id="data:0:123456",
            metrics_id="eval:metrics:0:000000",
            latest_n_pairs=50,
            pair_labels=["geography"],
            last_n_seconds=1800,  # Last 30 minutes
        )

        # Verify the request was made with all parameters
        expected_payload = {
            "seed_data_id": "data:0:123456",
            "metrics_id": "eval:metrics:0:000000",
            "pair_query": {
                "latest_n_pairs": 50,
                "pair_labels": ["geography"],
                "last_n_seconds": 1800,
            },
        }

        mock_post.assert_called_once_with(
            f"{self.base_url}/evaluate",
            json=expected_payload,
            headers=self.client.headers,
        )

    @patch("requests.post")
    def test_evaluate_synthetic_data(self, mock_post):
        # Setup
        mock_response = Mock()
        mock_response.json.return_value = {
            "eval_results_id": "eval:results:0:000000",
            "pair_count": 100,
            "scores": [],
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Test evaluating synthetic data
        result = self.client.evaluate(
            data_id="synth:0:123456",
            metrics_id="eval:metrics:0:000000",
            is_synthetic=True,
            latest_n_pairs=100,
        )

        # Verify synthetic_data_id is used instead of seed_data_id
        expected_payload = {
            "synthetic_data_id": "synth:0:123456",
            "metrics_id": "eval:metrics:0:000000",
            "pair_query": {"latest_n_pairs": 100},
        }

        mock_post.assert_called_once_with(
            f"{self.base_url}/evaluate",
            json=expected_payload,
            headers=self.client.headers,
        )

    @patch("requests.post")
    def test_evaluate_without_pair_query(self, mock_post):
        # Setup
        mock_response = Mock()
        mock_response.json.return_value = {
            "eval_results_id": "eval:results:0:000000",
            "pair_count": 150,
            "scores": [],
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Test without pair query parameters
        result = self.client.evaluate(
            data_id="data:0:123456", metrics_id="eval:metrics:0:000000"
        )

        # Verify no pair_query is included when no filtering parameters are provided
        expected_payload = {
            "seed_data_id": "data:0:123456",
            "metrics_id": "eval:metrics:0:000000",
        }

        mock_post.assert_called_once_with(
            f"{self.base_url}/evaluate",
            json=expected_payload,
            headers=self.client.headers,
        )

    @patch("requests.post")
    def test_augment_basic(self, mock_post):
        # Setup
        mock_response = Mock()
        mock_response.json.return_value = {
            "synthetic_data_id": "synth:0:123456",
            "created_at": "2024-01-01T00:00:00Z",
            "seed_data_size": 10,
            "synthetic_data_size": 30,
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Test basic augmentation
        result = self.client.augment(seed_data_id="data:0:123456", multiple=3)

        # Verify the request
        expected_payload = {"multiple": 3, "seed_data_id": "data:0:123456"}

        mock_post.assert_called_once_with(
            f"{self.base_url}/augment",
            json=expected_payload,
            headers=self.client.headers,
        )

        assert result["synthetic_data_id"] == "synth:0:123456"

    @patch("requests.post")
    def test_augment_with_all_params(self, mock_post):
        # Setup
        mock_response = Mock()
        mock_response.json.return_value = {
            "synthetic_data_id": "synth:0:123456",
            "created_at": "2024-01-01T00:00:00Z",
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Test augmentation with all parameters
        result = self.client.augment(
            seed_data_id="data:0:123456",
            multiple=2,
            eval_results_id="eval:results:0:000000",
            latest_n_pairs=50,
            pair_labels=["geography"],
            target_metric="accuracy",
        )

        # Verify the request includes all parameters
        expected_payload = {
            "multiple": 2,
            "seed_data_id": "data:0:123456",
            "eval_results_id": "eval:results:0:000000",
            "target_metric": "accuracy",
            "pair_query": {"latest_n_pairs": 50, "pair_labels": ["geography"]},
        }

        mock_post.assert_called_once_with(
            f"{self.base_url}/augment",
            json=expected_payload,
            headers=self.client.headers,
        )

    @patch("requests.post")
    def test_augment_minimal_params(self, mock_post):
        # Setup
        mock_response = Mock()
        mock_response.json.return_value = {"synthetic_data_id": "synth:0:123456"}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Test augmentation with minimal parameters
        result = self.client.augment(multiple=5)

        # Verify only multiple is included when other params are None
        expected_payload = {"multiple": 5}

        mock_post.assert_called_once_with(
            f"{self.base_url}/augment",
            json=expected_payload,
            headers=self.client.headers,
        )

    @patch("requests.post")
    def test_augment_error_handling(self, mock_post):
        # Setup
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "Internal server error"
        )
        mock_post.return_value = mock_response

        # Execute & Verify
        with self.assertRaises(requests.exceptions.HTTPError):
            self.client.augment(seed_data_id="data:0:123456", multiple=2)

    @patch("requests.post")
    def test_upload_pair_with_prompt(self, mock_post):
        # Setup
        test_api_key = "test-api-key"
        test_input = "This is a test input"
        test_output = "This is a test output"
        test_system_prompt = "You are a helpful assistant"
        test_pair_id = "test-pair-id"
        test_labels = ["label1", "label2"]
        test_dataset_id = "data:0:0000000"

        expected_url = "https://beta.getplum.ai/v1/data/seed/pair"
        expected_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": test_api_key,
        }
        expected_payload = {
            "input": test_input,
            "output": test_output,
            "system_prompt_template": test_system_prompt,
            "labels": test_labels,
            "id": test_pair_id,
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "dataset_id": test_dataset_id,
            "pair_id": test_pair_id,
        }

        mock_post.return_value = mock_response

        # Execute
        client = PlumClient(test_api_key)
        result = client.upload_pair_with_prompt(
            input_text=test_input,
            output_text=test_output,
            system_prompt_template=test_system_prompt,
            pair_id=test_pair_id,
            labels=test_labels,
        )

        # Verify
        mock_post.assert_called_once_with(
            expected_url, headers=expected_headers, json=expected_payload
        )
        assert isinstance(result, PairUploadResponse)
        assert result.dataset_id == test_dataset_id
        assert result.pair_id == test_pair_id

    @patch("requests.post")
    def test_upload_pair_with_prompt_minimal_params(self, mock_post):
        # Setup
        test_api_key = "test-api-key"
        test_input = "This is a test input"
        test_output = "This is a test output"
        test_system_prompt = "You are a helpful assistant"
        test_dataset_id = "data:0:0000000"
        auto_generated_pair_id = "auto-generated-id"

        expected_url = "https://beta.getplum.ai/v1/data/seed/pair"
        expected_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": test_api_key,
        }
        expected_payload = {
            "input": test_input,
            "output": test_output,
            "system_prompt_template": test_system_prompt,
            "labels": [],
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "dataset_id": test_dataset_id,
            "pair_id": auto_generated_pair_id,
        }

        mock_post.return_value = mock_response

        # Execute
        client = PlumClient(test_api_key)
        result = client.upload_pair_with_prompt(
            input_text=test_input,
            output_text=test_output,
            system_prompt_template=test_system_prompt,
        )

        # Verify
        mock_post.assert_called_once_with(
            expected_url, headers=expected_headers, json=expected_payload
        )
        assert isinstance(result, PairUploadResponse)
        assert result.dataset_id == test_dataset_id
        assert result.pair_id == auto_generated_pair_id

    @patch("requests.post")
    def test_upload_pair_with_prompt_error_handling(self, mock_post):
        # Setup
        test_api_key = "test-api-key"
        test_input = "This is a test input"
        test_output = "This is a test output"
        test_system_prompt = "You are a helpful assistant"

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "Invalid system prompt"
        )

        mock_post.return_value = mock_response

        # Execute & Verify
        client = PlumClient(test_api_key)
        with self.assertRaises(requests.exceptions.HTTPError):
            client.upload_pair_with_prompt(
                input_text=test_input,
                output_text=test_output,
                system_prompt_template=test_system_prompt,
            )

    @patch("requests.get")
    def test_get_dataset_success(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "dataset_123",
            "data": [
                {
                    "id": "pair_1",
                    "input": "test input",
                    "output": "test output",
                    "metadata": {
                        "created_at": "2023-01-01T00:00:00Z",
                        "labels": ["test"],
                    },
                }
            ],
            "system_prompt": "test system prompt",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        result = self.client.get_dataset("dataset_123")

        mock_get.assert_called_once_with(
            f"{self.base_url}/data/seed/dataset_123", headers=self.client.headers
        )
        self.assertEqual(result.id, "dataset_123")
        self.assertEqual(len(result.data), 1)
        self.assertEqual(result.data[0].id, "pair_1")
        self.assertEqual(result.data[0].input, "test input")
        self.assertEqual(result.data[0].output, "test output")
        self.assertEqual(result.system_prompt, "test system prompt")

    @patch("requests.get")
    def test_get_dataset_synthetic(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "dataset_456",
            "data": [],
            "system_prompt": "test system prompt",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        result = self.client.get_dataset("dataset_456", is_synthetic=True)

        mock_get.assert_called_once_with(
            f"{self.base_url}/data/synthetic/dataset_456", headers=self.client.headers
        )
        self.assertEqual(result.id, "dataset_456")

    @patch("requests.get")
    def test_get_dataset_failure(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError
        mock_get.return_value = mock_response

        with self.assertRaises(requests.exceptions.HTTPError):
            self.client.get_dataset("nonexistent_dataset")

    @patch("requests.get")
    def test_get_pair_success(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "dataset_123",
            "data": [
                {
                    "id": "pair_1",
                    "input": "test input 1",
                    "output": "test output 1",
                    "metadata": {
                        "created_at": "2023-01-01T00:00:00Z",
                        "labels": ["test"],
                    },
                },
                {
                    "id": "pair_2",
                    "input": "test input 2",
                    "output": "test output 2",
                    "metadata": {
                        "created_at": "2023-01-01T00:00:00Z",
                        "labels": [],
                    },
                },
            ],
            "system_prompt": "test system prompt",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        result = self.client.get_pair("dataset_123", "pair_2")

        mock_get.assert_called_once_with(
            f"{self.base_url}/data/seed/dataset_123", headers=self.client.headers
        )
        self.assertEqual(result.id, "pair_2")
        self.assertEqual(result.input, "test input 2")
        self.assertEqual(result.output, "test output 2")

    @patch("requests.get")
    def test_get_pair_not_found(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "dataset_123",
            "data": [
                {
                    "id": "pair_1",
                    "input": "test input 1",
                    "output": "test output 1",
                    "metadata": {
                        "created_at": "2023-01-01T00:00:00Z",
                        "labels": [],
                    },
                }
            ],
            "system_prompt": "test system prompt",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        with self.assertRaises(ValueError) as context:
            self.client.get_pair("dataset_123", "nonexistent_pair")

        self.assertIn(
            "Pair with ID 'nonexistent_pair' not found", str(context.exception)
        )
