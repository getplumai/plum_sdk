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
from plum_sdk import PlumClient, TrainingExample

# Initialize the SDK with your API key
api_key = "YOUR_API_KEY"
plum_client = PlumClient(api_key)

# Create training examples
training_examples = [
    TrainingExample(
        input="What is the capital of France?",
        output="The capital of France is Paris."
    ),
    TrainingExample(
        input="How do I make pasta?",
        output="1. Boil water\n2. Add salt\n3. Cook pasta until al dente"
    ),
    TrainingExample(
        id="custom_id_123",
        input="What is machine learning?",
        output="Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."
    )
]

# Define your system prompt
system_prompt = "You are a helpful assistant that provides accurate and concise answers."

# Upload the data
response = plum_client.upload_data(training_examples, system_prompt)
print(response)
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
    pair_label="geography"
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
    pair_label="geography",  # Only use pairs with this label
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

## API Reference

### PlumClient

#### Constructor
- `api_key` (str): Your Plum API key
- `base_url` (str, optional): Custom base URL for the Plum API

#### Methods
- `upload_data(training_examples: List[TrainingExample], system_prompt: str) -> UploadResponse`: 
  Uploads training examples and system prompt to Plum DB
  
- `upload_pair(dataset_id: str, input_text: str, output_text: str, pair_id: Optional[str] = None, labels: Optional[List[str]] = None) -> PairUploadResponse`:
  Adds a single input-output pair to an existing dataset

- `upload_pair_with_prompt(input_text: str, output_text: str, system_prompt_template: str, pair_id: Optional[str] = None, labels: Optional[List[str]] = None) -> PairUploadResponse`:
  Adds a single input-output pair to a dataset, creating the dataset if it doesn't exist

- `generate_metric_questions(system_prompt: str) -> MetricsQuestions`: 
  Automatically generates evaluation metric questions based on a system prompt

- `define_metric_questions(questions: List[str]) -> MetricsResponse`: 
  Defines custom evaluation metric questions

- `evaluate(data_id: str, metrics_id: str, latest_n_pairs: Optional[int] = None, pair_label: Optional[str] = None, is_synthetic: bool = False) -> EvaluationResponse`: 
  Evaluates uploaded data against defined metrics and returns detailed scoring results

- `augment(seed_data_id: Optional[str] = None, multiple: int = 1, eval_results_id: Optional[str] = None, latest_n_pairs: Optional[int] = None, pair_label: Optional[str] = None, target_metric: Optional[str] = None) -> dict`:
  Augments seed data to generate synthetic training examples

### Data Classes

#### TrainingExample
A dataclass representing a single training example:
- `input` (str): The input text
- `output` (str): The output text produced by your LLM
- `id` (Optional[str]): Optional custom identifier for the example

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

#### EvaluationResults
Contains evaluation results:
- `eval_results_id` (str): Unique identifier for the evaluation results
- `scores` (List[Dict]): Detailed scoring information including mean, median, standard deviation, and confidence intervals

