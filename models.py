from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingExample:
    input: str
    output: str
    id: Optional[str] = None


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
class IOPairMeta:
    """Metadata for an IO pair."""

    created_at: Optional[str] = None
    labels: Optional[List[str]] = None


@dataclass
class IOPair:
    """An input-output pair from a dataset."""

    id: str
    input: str
    output: str
    metadata: Optional[IOPairMeta] = None
    input_media: Optional[bytes] = None
    use_media_mime_type: Optional[str] = None
    human_critique: Optional[str] = None
    target_metric: Optional[str] = None


@dataclass
class Dataset:
    """A dataset containing IO pairs and metadata."""

    id: str
    data: List[IOPair]
    system_prompt: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class PairUploadResponse:
    """Response from uploading a pair to a dataset."""

    dataset_id: str
    pair_id: str


@dataclass
class MetricDefinition:
    """A single metric definition."""

    id: str
    name: str
    description: str


@dataclass
class DetailedMetricsResponse:
    """Detailed metrics response including definitions."""

    metrics_id: str
    definitions: List[MetricDefinition]
    system_prompt: Optional[str] = None
    metric_count: int = 0
    created_at: Optional[str] = None


@dataclass
class MetricsListResponse:
    """Response containing a list of all available metrics."""

    metrics: dict[str, DetailedMetricsResponse]
    total_count: int
