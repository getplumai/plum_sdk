# Plum SDK

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