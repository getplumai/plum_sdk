from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingExample:
    id: str
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


@dataclass
class PairUploadResponse:
    """Response from uploading a pair to a dataset."""

    dataset_id: str
    pair_id: str
