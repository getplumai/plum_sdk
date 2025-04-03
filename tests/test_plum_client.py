import unittest
from unittest.mock import Mock, patch
import requests
import pytest
from plum_sdk import PlumClient, TrainingExample
from plum_sdk.models import (
    PairUploadResponse,
    TrainingExample,
    UploadResponse,
    MetricsQuestions,
    MetricsResponse,
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
        with pytest.raises(requests.exceptions.HTTPError, match="Dataset not found"):
            client.upload_pair(
                dataset_id=test_dataset_id,
                input_text=test_input,
                output_text=test_output,
            )
