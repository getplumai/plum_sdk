# Plum SDK

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/plum-sdk.svg)](https://badge.fury.io/py/plum-sdk)

Python SDK for [Plum AI](https://getplum.ai).

## Installation

```bash
pip install plum-sdk
```

## Usage

The Plum SDK allows you to upload training examples, generate and define metric questions, and evaluate your LLM's performance.

### Basic Usage

```python
from plum_sdk import PlumClient, IOPair

# Initialize the SDK with your API key
api_key = "YOUR_API_KEY"
plum_client = PlumClient(api_key)

# Or initialize with a custom base URL
# plum_client = PlumClient(api_key, base_url="https://custom.api.url/v1")

# Create training examples
training_examples = [
    IOPair(
        input="What is the capital of France?",
        output="The capital of France is Paris."
    ),
    IOPair(
        input="How do I make pasta?",
        output="1. Boil water\n2. Add salt\n3. Cook pasta until al dente"
    ),
    IOPair(
        id="custom_id_123",
        input="What is machine learning?",
        output="Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."
    )
]

# Define your system prompt
system_prompt = "You are a helpful assistant that provides accurate and concise answers."

# Upload the data
response = plum_client.upload_data(training_examples, system_prompt)
print(f"Dataset uploaded with ID: {response.id}")
```

### Updating an Existing Dataset

You can also upload training examples to update an existing dataset by providing the dataset ID:

```python
# Update an existing dataset with new training examples
existing_dataset_id = "data:0:123456"  # ID from a previous dataset
new_training_examples = [
    IOPair(
        input="What is the capital of Italy?",
        output="The capital of Italy is Rome."
    ),
    IOPair(
        input="How do you say 'hello' in Spanish?",
        output="'Hello' in Spanish is 'Hola'."
    )
]

# This will overwrite the existing dataset with the new data
response = plum_client.upload_data(
    training_examples=new_training_examples, 
    system_prompt=system_prompt,
    dataset_id=existing_dataset_id
)
print(f"Dataset updated with ID: {response.id}")
```

### Adding Individual Examples to an Existing Dataset

You can add additional training examples to an existing dataset:

```python
# Add a single example to an existing dataset
dataset_id = "data:0:123456" # ID from previous upload_data response
response = plum_client.upload_pair(
    dataset_id=dataset_id,
    input_text="What is the tallest mountain in the world?",
    output_text="Mount Everest is the tallest mountain in the world, with a height of 8,848.86 meters (29,031.7 feet).",
    labels=["geography", "mountains"]  # Optional labels for categorization
)
print(f"Added pair with ID: {response.pair_id}")
```

### Adding Examples with System Prompt (Auto-dataset Creation)

If you want to add a single example but don't have an existing dataset ID, you can use `upload_pair_with_prompt`. This method will either find an existing dataset with the same system prompt or create a new one:

```python
# Add a single example with a system prompt - will auto-create or find matching dataset
response = plum_client.upload_pair_with_prompt(
    input_text="What is the capital of Japan?",
    output_text="The capital of Japan is Tokyo.",
    system_prompt_template="You are a helpful assistant that provides accurate and concise answers.",
    labels=["geography", "capitals"]  # Optional labels
)
print(f"Added pair with ID: {response.pair_id} to dataset: {response.dataset_id}")
```

### Generating and Evaluating with Metrics

```python
# Generate evaluation metrics based on your system prompt
metrics_response = plum_client.generate_metric_questions(system_prompt)
print(f"Generated metrics with ID: {metrics_response.metrics_id}")

# Evaluate your dataset
evaluation_response = plum_client.evaluate(
    data_id=response.id,  # Dataset ID from upload_data response
    metrics_id=metrics_response.metrics_id
)
print(f"Evaluation completed with ID: {evaluation_response.eval_results_id}")
```

### Advanced Evaluation with Filtering

You can filter which pairs to evaluate using `pair_query` parameters:

```python
# Evaluate only the latest 50 pairs
evaluation_response = plum_client.evaluate(
    data_id=dataset_id,
    metrics_id=metrics_id,
    latest_n_pairs=50
)

# Evaluate only pairs with specific labels
evaluation_response = plum_client.evaluate(
    data_id=dataset_id,
    metrics_id=metrics_id,
    pair_labels=["geography"]
)

# Evaluate only pairs created in the last hour (3600 seconds)
evaluation_response = plum_client.evaluate(
    data_id=dataset_id,
    metrics_id=metrics_id,
    last_n_seconds=3600
)

# Combine multiple filters
evaluation_response = plum_client.evaluate(
    data_id=dataset_id,
    metrics_id=metrics_id,
    latest_n_pairs=50,
    pair_labels=["geography", "capitals"], # Only pairs tagged with both "geography" AND "capitals" labels
    last_n_seconds=1800  # Last 30 minutes
)

# Evaluate synthetic data instead of seed data
evaluation_response = plum_client.evaluate(
    data_id=synthetic_data_id,
    metrics_id=metrics_id,
    is_synthetic=True,
    latest_n_pairs=100
)
```

### Data Augmentation

Generate synthetic training examples from your seed data:

```python
# Basic augmentation - generates 3x the original dataset size
augment_response = plum_client.augment(
    seed_data_id=dataset_id,
    multiple=3
)
print(f"Generated synthetic data with ID: {augment_response['synthetic_data_id']}")

# Advanced augmentation with filtering and target metric
augment_response = plum_client.augment(
    seed_data_id=dataset_id,
    multiple=2,
    eval_results_id=evaluation_response.eval_results_id,
    latest_n_pairs=50,  # Only use latest 50 pairs for augmentation
    pair_labels=["geography"],  # Only use pairs with these labels
    target_metric="accuracy"  # Target specific metric for redrafting
)
```

### Error Handling

The SDK will raise exceptions for non-200 responses:

```python
from plum_sdk import PlumClient
import requests

try:
    plum_client = PlumClient(api_key="YOUR_API_KEY")
    response = plum_client.upload_data(training_examples, system_prompt)
    print(response)
except requests.exceptions.HTTPError as e:
    print(f"Error uploading data: {e}")
```

### Data Retrieval

Retrieve datasets and individual pairs from Plum:

```python
# Get a complete dataset with all its pairs
dataset = plum_client.get_dataset(dataset_id="data:0:123456")
print(f"Dataset {dataset.id} contains {len(dataset.data)} pairs")
print(f"System prompt: {dataset.system_prompt}")

# Iterate through all pairs in the dataset
for pair in dataset.data:
    print(f"Pair {pair.id}: {pair.input[:50]}...")
    if pair.metadata and pair.metadata.labels:
        print(f"  Labels: {pair.metadata.labels}")

# Get a synthetic dataset instead of seed data
synthetic_dataset = plum_client.get_dataset(
    dataset_id="synthetic:0:789012", 
    is_synthetic=True
)

# Get a specific pair from a dataset
pair = plum_client.get_pair(
    dataset_id="data:0:123456",
    pair_id="pair_abc123"
)
print(f"Input: {pair.input}")
print(f"Output: {pair.output}")
if pair.metadata:
    print(f"Created at: {pair.metadata.created_at}")
    print(f"Labels: {pair.metadata.labels}")
```

### Metrics Management

List and retrieve evaluation metrics:

```python
# List all available metrics
metrics_list = plum_client.list_metrics()
print(f"Total metrics available: {metrics_list.total_count}")

# Browse through all metrics
for metrics_id, metric_details in metrics_list.metrics.items():
    print(f"\nMetrics ID: {metrics_id}")
    print(f"Created at: {metric_details.created_at}")
    print(f"Number of questions: {metric_details.metric_count}")
    
    # Show each metric question
    for definition in metric_details.definitions:
        print(f"  - {definition.name}: {definition.description}")

# Get detailed information about a specific metric
metric_details = plum_client.get_metric(metrics_id="metrics:0:456789")
print(f"Metric {metric_details.metrics_id} has {metric_details.metric_count} questions")
if metric_details.system_prompt:
    print(f"Associated system prompt: {metric_details.system_prompt}")

# Show all metric definitions
for definition in metric_details.definitions:
    print(f"Question ID: {definition.id}")
    print(f"Name: {definition.name}")
    print(f"Description: {definition.description}")
```

## API Reference

### PlumClient

#### Constructor
- `api_key` (str): Your Plum API key
- `base_url` (str, optional): Custom base URL for the Plum API (defaults to "https://beta.getplum.ai/v1")

#### Methods
- `upload_data(training_examples: List[IOPair], system_prompt: str, dataset_id: Optional[str] = None) -> UploadResponse`: 
  Uploads training examples and system prompt to create a new dataset or update an existing one. If dataset_id is provided, overwrites the existing dataset; otherwise creates a new dataset.
  
- `upload_pair(dataset_id: str, input_text: str, output_text: str, pair_id: Optional[str] = None, labels: Optional[List[str]] = None) -> PairUploadResponse`:
  Adds a single input-output pair to an existing dataset

- `upload_pair_with_prompt(input_text: str, output_text: str, system_prompt_template: str, pair_id: Optional[str] = None, labels: Optional[List[str]] = None) -> PairUploadResponse`:
  Adds a single input-output pair to a dataset, creating the dataset if it doesn't exist

- `generate_metric_questions(system_prompt: str) -> MetricsQuestions`: 
  Automatically generates evaluation metric questions based on a system prompt

- `define_metric_questions(questions: List[str]) -> MetricsResponse`: 
  Defines custom evaluation metric questions

- `evaluate(data_id: str, metrics_id: str, latest_n_pairs: Optional[int] = None, pair_labels: Optional[List[str]] = None, last_n_seconds: Optional[int] = None, is_synthetic: bool = False) -> EvaluationResponse`: 
  Evaluates uploaded data against defined metrics and returns detailed scoring results

- `augment(seed_data_id: Optional[str] = None, multiple: int = 1, eval_results_id: Optional[str] = None, latest_n_pairs: Optional[int] = None, pair_labels: Optional[List[str]] = None, target_metric: Optional[str] = None) -> dict`:
  Augments seed data to generate synthetic training examples

- `get_dataset(dataset_id: str, is_synthetic: bool = False) -> Dataset`:
  Retrieves a complete dataset with all its pairs by ID

- `get_pair(dataset_id: str, pair_id: str, is_synthetic: bool = False) -> IOPair`:
  Retrieves a specific pair from a dataset by its ID

- `list_metrics() -> MetricsListResponse`:
  Lists all available evaluation metrics with their definitions

- `get_metric(metrics_id: str) -> DetailedMetricsResponse`:
  Retrieves detailed information about a specific metric by ID

### Data Classes

#### IOPair
A dataclass representing a single example of your application's interaction with an LLM:
- `input` (str): The input text
- `output` (str): The output text produced by your LLM
- `id` (Optional[str]): Optional custom identifier for the example

#### UploadResponse
Response from uploading training data:
- `id` (str): Unique identifier for the created dataset

#### PairUploadResponse
Response from uploading a pair to a dataset:
- `dataset_id` (str): ID of the dataset the pair was added to
- `pair_id` (str): Unique identifier for the uploaded pair

#### MetricsQuestions
Contains generated evaluation metrics:
- `metrics_id` (str): Unique identifier for the metrics
- `definitions` (List[str]): List of generated metric questions

#### MetricsResponse
Response from defining custom metrics:
- `metrics_id` (str): Unique identifier for the defined metrics

#### EvaluationResponse
Contains evaluation results:
- `eval_results_id` (str): Unique identifier for the evaluation results
- `scores` (List[MetricScore]): Detailed scoring information for each metric
- `pair_count` (int): Number of pairs that were evaluated
- `dataset_id` (Optional[str]): ID of the dataset that was evaluated
- `created_at` (Optional[str]): Timestamp when the evaluation was created

#### MetricScore
Contains detailed scoring information for a single metric:
- `metric` (str): Name of the metric
- `mean_score` (float): Average score across all evaluated pairs
- `std_dev` (float): Standard deviation of scores
- `ci_low` (float): Lower bound of confidence interval
- `ci_high` (float): Upper bound of confidence interval
- `ci_confidence` (float): Confidence level (e.g., 0.95 for 95%)
- `median_score` (float): Median score across all evaluated pairs
- `min_score` (float): Minimum score observed
- `max_score` (float): Maximum score observed
- `lowest_scoring_pairs` (List[ScoringPair]): Pairs that received the lowest scores

#### ScoringPair
Contains information about a specific scoring result:
- `pair_id` (str): Unique identifier for the pair
- `score_reason` (str): Explanation of why the pair received its score

#### Dataset
Contains a complete dataset with all its pairs:
- `id` (str): Unique identifier for the dataset
- `data` (List[IOPair]): List of all input-output pairs in the dataset
- `system_prompt` (Optional[str]): The system prompt associated with the dataset
- `created_at` (Optional[str]): Timestamp when the dataset was created

#### IOPair
Represents a single input-output pair:
- `id` (str): Unique identifier for the pair
- `input` (str): The input text
- `output` (str): The output text
- `metadata` (Optional[IOPairMeta]): Additional metadata about the pair
- `input_media` (Optional[bytes]): Optional media content
- `use_media_mime_type` (Optional[str]): MIME type for media content
- `human_critique` (Optional[str]): Human feedback on the pair
- `target_metric` (Optional[str]): Target metric for evaluation

#### IOPairMeta
Metadata for an input-output pair:
- `created_at` (Optional[str]): Timestamp when the pair was created
- `labels` (Optional[List[str]]): List of labels associated with the pair

#### MetricsListResponse
Response containing all available metrics:
- `metrics` (Dict[str, DetailedMetricsResponse]): Dictionary mapping metric IDs to detailed metric information
- `total_count` (int): Total number of available metrics

#### DetailedMetricsResponse
Detailed information about a specific metric:
- `metrics_id` (str): Unique identifier for the metric
- `definitions` (List[MetricDefinition]): List of all metric questions/definitions
- `system_prompt` (Optional[str]): System prompt associated with the metric
- `metric_count` (int): Number of questions in the metric
- `created_at` (Optional[str]): Timestamp when the metric was created

#### MetricDefinition
Individual metric question definition:
- `id` (str): Unique identifier for the metric question
- `name` (str): Display name of the metric question
- `description` (str): Detailed description of what the metric evaluates

