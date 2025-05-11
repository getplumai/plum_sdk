import requests
from typing import List, Optional
from plum_sdk.models import (
    TrainingExample,
    UploadResponse,
    MetricsQuestions,
    MetricsResponse,
    EvaluationResponse,
    PairUploadResponse,
)


class PlumClient:
    def __init__(self, api_key: str, base_url: str = "https://beta.getplum.ai/v1"):
        """
        Initialize a new PlumClient instance.

        Args:
            api_key: Your Plum API authentication key
            base_url: The base URL for the Plum API (defaults to "https://beta.getplum.ai/v1")
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"{self.api_key}",
        }

    def upload_data(
        self, training_examples: List[TrainingExample], system_prompt: str
    ) -> UploadResponse:
        """
        Upload training examples with a system prompt to create a new dataset.

        Args:
            training_examples: A list of TrainingExample objects containing input-output pairs
            system_prompt: The system prompt to use with the training examples

        Returns:
            UploadResponse object containing information about the uploaded dataset

        Raises:
            requests.HTTPError: If the request to the Plum API fails
        """
        url = f"{self.base_url}/data/seed"

        data = []
        for example in training_examples:
            pair = {"input": example.input, "output": example.output}
            if hasattr(example, "id") and example.id:
                pair["id"] = example.id
            data.append(pair)

        payload = {"data": data, "system_prompt": system_prompt}

        response = requests.post(url, json=payload, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            return UploadResponse(**data)
        else:
            response.raise_for_status()

    def upload_pair(
        self,
        dataset_id: str,
        input_text: str,
        output_text: str,
        pair_id: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> PairUploadResponse:
        """
        Upload a single input-output pair to an existing seed dataset.

        Args:
            dataset_id: ID of the existing seed dataset to add the pair to
            input_text: The user prompt/input text
            output_text: The output/response text
            pair_id: Optional custom ID for the pair (will be auto-generated if not provided)
            labels: Optional list of labels to associate with this pair

        Returns:
            Dict containing the pair_id and corpus_id

        Raises:
            requests.HTTPError: If the request fails
        """
        if labels is None:
            labels = []

        endpoint = f"{self.base_url}/data/seed/{dataset_id}/pair"

        payload = {"input": input_text, "output": output_text, "labels": labels}

        if pair_id:
            payload["id"] = pair_id

        response = requests.post(endpoint, headers=self.headers, json=payload)

        response.raise_for_status()
        response_data = response.json()
        return PairUploadResponse(
            dataset_id=response_data["dataset_id"], pair_id=response_data["pair_id"]
        )

    def generate_metric_questions(self, system_prompt: str) -> MetricsQuestions:
        """
        Generate evaluation metric questions based on a system prompt.

        Args:
            system_prompt: The system prompt to generate evaluation questions for

        Returns:
            MetricsQuestions object containing the generated questions

        Raises:
            requests.HTTPError: If the request to the Plum API fails
        """
        url = f"{self.base_url}/questions"

        payload = {"system_prompt": system_prompt}

        response = requests.post(url, json=payload, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            return MetricsQuestions(**data)
        else:
            response.raise_for_status()

    def define_metric_questions(self, metrics: List[str]) -> MetricsResponse:
        """
        Define custom evaluation metric questions.

        Args:
            metrics: A list of strings describing the evaluation metrics

        Returns:
            MetricsResponse object containing information about the defined metrics

        Raises:
            requests.HTTPError: If the request to the Plum API fails
        """
        url = f"{self.base_url}/specify_questions"

        payload = {"metrics": metrics}

        response = requests.post(url, json=payload, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            return MetricsResponse(**data)
        else:
            response.raise_for_status()

    def evaluate(self, data_id: str, metrics_id: str) -> EvaluationResponse:
        """
        Evaluate a dataset using specified metrics.

        Args:
            data_id: The ID of the dataset to evaluate
            metrics_id: The ID of the metrics to use for evaluation

        Returns:
            EvaluationResponse object containing the evaluation results

        Raises:
            requests.HTTPError: If the request to the Plum API fails
        """
        url = f"{self.base_url}/evaluate"

        payload = {"seed_data_id": data_id, "metrics_id": metrics_id}

        response = requests.post(url, json=payload, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            return EvaluationResponse(**data)
        else:
            response.raise_for_status()
