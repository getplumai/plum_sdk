import requests
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TrainingExample:
    input: str
    output: str

@dataclass
class UploadResponse:
    id: str

@dataclass
class MetricsQuestions:
    metrics_id: str
    definitions: List[str]

@dataclass
class Question:
    id: str
    input: str
    status: str
    created_at: str
    updated_at: str
    prompt: Optional[str] = None
    stream_id: Optional[str] = None

@dataclass
class MetricsResponse:
    metrics_id: str

@dataclass
class ScoringPair:
    pair_id: str
    score_reason: str

@dataclass
class MetricScore:
    metric: str
    mean_score: float
    std_dev: float
    ci_low: float
    ci_high: float
    ci_confidence: float
    median_score: float
    min_score: float
    max_score: float
    lowest_scoring_pairs: List[ScoringPair]

@dataclass
class EvaluationResponse:
    eval_results_id: str
    scores: List[MetricScore]
    pair_count: int

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
