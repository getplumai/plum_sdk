import requests
from typing import List, Optional
from .models import TrainingExample, UploadResponse, MetricsQuestions, MetricsResponse, EvaluationResponse, PairUploadResponse


class PlumClient:
    def __init__(self, api_key: str, base_url: str = "https://beta.getplum.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"{self.api_key}"
        }

    def upload_data(self, training_examples: List[TrainingExample], system_prompt: str) -> UploadResponse:
        url = f"{self.base_url}/data/seed"

        data = [
            {"input": example.input, "output": example.output}
            for example in training_examples
        ]

        payload = {
            "data": data,
            "system_prompt": system_prompt
        }

        response = requests.post(url, json=payload, headers=self.headers)

        if response.status_code == 200:
             data = response.json()
             return UploadResponse(**data)
        else:
            response.raise_for_status()
    
    def upload_pair(self, 
                   dataset_id: str, 
                   input_text: str, 
                   output_text: str, 
                   pair_id: Optional[str] = None,
                   labels: Optional[List[str]] = None) -> PairUploadResponse:
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
        
        payload = {
            "input": input_text,
            "output": output_text,
            "labels": labels
        }
        
        if pair_id:
            payload["id"] = pair_id
            
        response = requests.post(
            endpoint,
            headers=self.headers,
            json=payload
        )
        
        response.raise_for_status()
        response_data = response.json()
        return PairUploadResponse(
            dataset_id=response_data["dataset_id"],
            pair_id=response_data["pair_id"]
        )
    
    def generate_metric_questions(self, system_prompt: str) -> MetricsQuestions:
        url = f"{self.base_url}/questions"

        payload = {
            "system_prompt": system_prompt
        }

        response = requests.post(url, json=payload, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            return MetricsQuestions(**data)
        else:
            response.raise_for_status()

    def define_metric_questions(self, metrics: List[str]) -> MetricsResponse:
        url = f"{self.base_url}/specify_questions"

        payload = {
            "metrics": metrics
        }

        response = requests.post(url, json=payload, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            return MetricsResponse(**data)
        else:
            response.raise_for_status()

    def evaluate(self, data_id: str, metrics_id: str) -> EvaluationResponse:
        url = f"{self.base_url}/evaluate"

        payload = {
            "seed_data_id": data_id,
            "metrics_id": metrics_id
        }

        response = requests.post(url, json=payload, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            return EvaluationResponse(**data)
        else:
            response.raise_for_status()
