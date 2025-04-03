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
  
- `generate_metric_questions(system_prompt: str) -> MetricsQuestions`: 
  Automatically generates evaluation metric questions based on a system prompt

- `define_metric_questions(questions: List[str]) -> MetricsResponse`: 
  Defines custom evaluation metric questions

- `evaluate(metrics_id: str, data_id: str) -> EvaluationResults`: 
  Evaluates uploaded data against defined metrics and returns detailed scoring results

### Data Classes

#### TrainingExample
A dataclass representing a single training example:
- `input` (str): The input text
- `output` (str): The output text produced by your LLM

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

