import requests
from typing import List, Optional, Dict, Any, Type, get_type_hints
from dataclasses import fields, is_dataclass
from .models import (
    TrainingExample,
    UploadResponse,
    MetricsQuestions,
    MetricsResponse,
    EvaluationResponse,
    MetricScore,
    ScoringPair,
    PairUploadResponse,
    IOPair,
    IOPairMeta,
    Dataset,
    MetricsListResponse,
    DetailedMetricsResponse,
    MetricDefinition,
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

    def _filter_response_for_dataclass(self, response_data: Dict[str, Any], target_class: Type) -> Dict[str, Any]:
        """
        Filter API response data to only include fields that exist in the target dataclass.
        
        Args:
            response_data: Raw response data from API
            target_class: Target dataclass to filter fields for
            
        Returns:
            Filtered dictionary containing only fields that match the dataclass
        """
        if not is_dataclass(target_class):
            return response_data
            
        # Get all field names from the dataclass
        dataclass_fields = {field.name for field in fields(target_class)}
        
        # Filter the response to only include fields that exist in the dataclass
        filtered_data = {k: v for k, v in response_data.items() if k in dataclass_fields}
        
        return filtered_data

    def _filter_nested_objects(self, data_list: List[Dict[str, Any]], target_class: Type) -> List[Dict[str, Any]]:
        """
        Filter a list of nested objects to only include fields that exist in the target dataclass.
        
        Args:
            data_list: List of dictionaries to filter
            target_class: Target dataclass to filter fields for
            
        Returns:
            List of filtered dictionaries
        """
        return [self._filter_response_for_dataclass(item, target_class) for item in data_list]

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
            filtered_data = self._filter_response_for_dataclass(data, UploadResponse)
            return UploadResponse(**filtered_data)
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
        filtered_data = self._filter_response_for_dataclass(response_data, PairUploadResponse)
        return PairUploadResponse(**filtered_data)

    def upload_pair_with_prompt(
        self,
        input_text: str,
        output_text: str,
        system_prompt_template: str,
        pair_id: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> PairUploadResponse:
        """
        Upload a single input-output pair with a system prompt template.

        If a dataset with the same system prompt already exists, the pair will be added to that dataset.
        If no such dataset exists, a new dataset will be created with the provided system prompt.

        Args:
            input_text: The user prompt/input text
            output_text: The output/response text
            system_prompt_template: The system prompt template for the dataset
            pair_id: Optional custom ID for the pair (will be auto-generated if not provided)
            labels: Optional list of labels to associate with this pair

        Returns:
            PairUploadResponse containing the pair_id and dataset_id (existing or newly created)

        Raises:
            requests.HTTPError: If the request fails
        """
        if labels is None:
            labels = []

        endpoint = f"{self.base_url}/data/seed/pair"

        payload = {
            "input": input_text,
            "output": output_text,
            "system_prompt_template": system_prompt_template,
            "labels": labels,
        }

        if pair_id:
            payload["id"] = pair_id

        response = requests.post(endpoint, headers=self.headers, json=payload)

        response.raise_for_status()
        response_data = response.json()
        filtered_data = self._filter_response_for_dataclass(response_data, PairUploadResponse)
        return PairUploadResponse(**filtered_data)

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
            filtered_data = self._filter_response_for_dataclass(data, MetricsQuestions)
            return MetricsQuestions(**filtered_data)
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
            filtered_data = self._filter_response_for_dataclass(data, MetricsResponse)
            return MetricsResponse(**filtered_data)
        else:
            response.raise_for_status()

    def evaluate(
        self,
        data_id: str,
        metrics_id: str,
        latest_n_pairs: Optional[int] = None,
        pair_labels: Optional[List[str]] = None,
        last_n_seconds: Optional[int] = None,
        is_synthetic: bool = False,
    ) -> EvaluationResponse:
        """
        Evaluate a dataset using specified metrics.

        Args:
            data_id: The ID of the dataset to evaluate
            metrics_id: The ID of the metrics to use for evaluation
            latest_n_pairs: Maximum number of latest pairs to include (defaults to 150 if not provided)
            pair_labels: Filter pairs by labels (optional list of strings)
            last_n_seconds: Filter pairs created in the last N seconds (optional)
            is_synthetic: Whether the data_id refers to synthetic data (default: False for seed data)

        Returns:
            EvaluationResponse object containing the evaluation results

        Raises:
            requests.HTTPError: If the request to the Plum API fails
        """
        url = f"{self.base_url}/evaluate"

        if is_synthetic:
            payload = {"synthetic_data_id": data_id, "metrics_id": metrics_id}
        else:
            payload = {"seed_data_id": data_id, "metrics_id": metrics_id}

        # Add pair_query if any filtering parameters are provided
        if (
            latest_n_pairs is not None
            or pair_labels is not None
            or last_n_seconds is not None
        ):
            pair_query = {}
            if latest_n_pairs is not None:
                pair_query["latest_n_pairs"] = latest_n_pairs
            if pair_labels is not None:
                pair_query["pair_labels"] = pair_labels
            if last_n_seconds is not None:
                pair_query["last_n_seconds"] = last_n_seconds
            payload["pair_query"] = pair_query

        response = requests.post(url, json=payload, headers=self.headers)

        if response.status_code == 200:
            data = response.json()

            # Convert scores data to MetricScore objects if present
            if "scores" in data and data["scores"]:
                scores = []
                for score_data in data["scores"]:
                    # Filter score_data for MetricScore fields
                    filtered_score_data = self._filter_response_for_dataclass(score_data, MetricScore)
                    
                    # Convert lowest_scoring_pairs to ScoringPair objects
                    scoring_pairs = []
                    if "lowest_scoring_pairs" in score_data:
                        for pair_data in score_data["lowest_scoring_pairs"]:
                            filtered_pair_data = self._filter_response_for_dataclass(pair_data, ScoringPair)
                            scoring_pairs.append(ScoringPair(**filtered_pair_data))

                    # Create MetricScore with ScoringPair objects
                    filtered_score_data["lowest_scoring_pairs"] = scoring_pairs
                    scores.append(MetricScore(**filtered_score_data))

                data["scores"] = scores

            filtered_data = self._filter_response_for_dataclass(data, EvaluationResponse)
            return EvaluationResponse(**filtered_data)
        else:
            response.raise_for_status()

    def augment(
        self,
        seed_data_id: Optional[str] = None,
        multiple: int = 1,
        eval_results_id: Optional[str] = None,
        latest_n_pairs: Optional[int] = None,
        pair_labels: Optional[List[str]] = None,
        target_metric: Optional[str] = None,
    ) -> dict:
        """
        Augment seed data to generate synthetic data.

        Args:
            seed_data_id: ID of seed dataset to augment (will use latest if not provided)
            multiple: Number of synthetic examples to generate per seed example (max 50)
            eval_results_id: ID of evaluation results to use for target metric (will use latest if not provided)
            latest_n_pairs: Maximum number of latest pairs to include (defaults to 150 if not provided)
            pair_labels: Filter pairs by labels (optional list of strings)
            target_metric: Target metric for redrafting synthetic data (will use lowest scoring metric if not provided)

        Returns:
            Dict containing augmentation results including synthetic_data_id

        Raises:
            requests.HTTPError: If the request to the Plum API fails
        """
        url = f"{self.base_url}/augment"

        payload = {"multiple": multiple}

        if seed_data_id is not None:
            payload["seed_data_id"] = seed_data_id
        if eval_results_id is not None:
            payload["eval_results_id"] = eval_results_id
        if target_metric is not None:
            payload["target_metric"] = target_metric

        # Add pair_query if any filtering parameters are provided
        if latest_n_pairs is not None or pair_labels is not None:
            pair_query = {}
            if latest_n_pairs is not None:
                pair_query["latest_n_pairs"] = latest_n_pairs
            if pair_labels is not None:
                pair_query["pair_labels"] = pair_labels
            payload["pair_query"] = pair_query

        response = requests.post(url, json=payload, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

    def get_dataset(self, dataset_id: str, is_synthetic: bool = False) -> Dataset:
        """
        Get a dataset by ID.

        Args:
            dataset_id: The ID of the dataset to retrieve
            is_synthetic: Whether the dataset is synthetic data (default: False for seed data)

        Returns:
            Dataset object containing the dataset information and all pairs

        Raises:
            requests.HTTPError: If the request fails
        """
        if is_synthetic:
            endpoint = f"{self.base_url}/data/synthetic/{dataset_id}"
        else:
            endpoint = f"{self.base_url}/data/seed/{dataset_id}"

        response = requests.get(endpoint, headers=self.headers)
        response.raise_for_status()

        data = response.json()

        # Convert the response data to our model format
        pairs = []
        for pair_data in data.get("data", []):
            # Filter pair data for IOPair fields
            filtered_pair_data = self._filter_response_for_dataclass(pair_data, IOPair)
            
            metadata = None
            if "metadata" in pair_data:
                # Filter metadata for IOPairMeta fields
                filtered_metadata = self._filter_response_for_dataclass(pair_data["metadata"], IOPairMeta)
                metadata = IOPairMeta(**filtered_metadata)

            # Remove metadata from filtered_pair_data if it exists, since we handle it separately
            filtered_pair_data.pop("metadata", None)
            filtered_pair_data["metadata"] = metadata
            
            pairs.append(IOPair(**filtered_pair_data))

        # Filter top-level data for Dataset fields
        filtered_data = self._filter_response_for_dataclass(data, Dataset)
        filtered_data["data"] = pairs
        
        return Dataset(**filtered_data)

    def get_pair(
        self, dataset_id: str, pair_id: str, is_synthetic: bool = False
    ) -> IOPair:
        """
        Get a specific pair from a dataset.

        Args:
            dataset_id: The ID of the dataset containing the pair
            pair_id: The ID of the specific pair to retrieve
            is_synthetic: Whether the dataset is synthetic data (default: False for seed data)

        Returns:
            IOPair object containing the pair information

        Raises:
            requests.HTTPError: If the request fails
            ValueError: If the pair is not found in the dataset
        """
        dataset = self.get_dataset(dataset_id, is_synthetic)

        # Find the specific pair by ID
        for pair in dataset.data:
            if pair.id == pair_id:
                return pair

        raise ValueError(
            f"Pair with ID '{pair_id}' not found in dataset '{dataset_id}'"
        )

    def list_metrics(self) -> MetricsListResponse:
        """
        List all available evaluation metrics.

        Returns:
            MetricsListResponse object containing all available metrics with their definitions

        Raises:
            requests.HTTPError: If the request to the Plum API fails
        """
        url = f"{self.base_url}/list_questions"

        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            data = response.json()

            # Convert the response to our model format
            metrics_dict = {}
            for metric_id, metric_data in data.get("metrics", {}).items():
                # Filter metric_data for DetailedMetricsResponse fields
                filtered_metric_data = self._filter_response_for_dataclass(metric_data, DetailedMetricsResponse)
                
                # Convert definitions list to MetricDefinition objects
                definitions = []
                for i, definition in enumerate(metric_data.get("definitions", [])):
                    # Handle different formats of definition data
                    if isinstance(definition, dict):
                        filtered_definition = self._filter_response_for_dataclass(definition, MetricDefinition)
                        # Provide defaults for missing required fields
                        if "id" not in filtered_definition:
                            filtered_definition["id"] = f"metric_{i}"
                        if "name" not in filtered_definition:
                            filtered_definition["name"] = f"Metric {i+1}"
                        if "description" not in filtered_definition:
                            filtered_definition["description"] = definition.get("text", str(definition))
                        definitions.append(MetricDefinition(**filtered_definition))
                    else:
                        # If it's a string, use it as the description
                        definitions.append(
                            MetricDefinition(
                                id=f"metric_{i}",
                                name=f"Metric {i+1}",
                                description=str(definition),
                            )
                        )

                # Override with processed definitions and ensure required fields
                filtered_metric_data["definitions"] = definitions
                if "metrics_id" not in filtered_metric_data:
                    filtered_metric_data["metrics_id"] = metric_id
                if "metric_count" not in filtered_metric_data:
                    filtered_metric_data["metric_count"] = len(definitions)
                    
                metrics_dict[metric_id] = DetailedMetricsResponse(**filtered_metric_data)

            # Filter top-level response for MetricsListResponse fields
            filtered_response = self._filter_response_for_dataclass(data, MetricsListResponse)
            filtered_response["metrics"] = metrics_dict
            if "total_count" not in filtered_response:
                filtered_response["total_count"] = len(metrics_dict)
                
            return MetricsListResponse(**filtered_response)
        else:
            response.raise_for_status()

    def get_metric(self, metrics_id: str) -> DetailedMetricsResponse:
        """
        Get a specific metric definition by ID.

        Args:
            metrics_id: The ID of the metric to retrieve

        Returns:
            DetailedMetricsResponse object containing the metric definition and all its questions

        Raises:
            requests.HTTPError: If the request to the Plum API fails
        """
        url = f"{self.base_url}/question/{metrics_id}"

        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            data = response.json()

            # Convert definitions list to MetricDefinition objects
            definitions = []
            for i, definition in enumerate(data.get("definitions", [])):
                # Handle different formats of definition data
                if isinstance(definition, dict):
                    filtered_definition = self._filter_response_for_dataclass(definition, MetricDefinition)
                    # Provide defaults for missing required fields
                    if "id" not in filtered_definition:
                        filtered_definition["id"] = f"metric_{i}"
                    if "name" not in filtered_definition:
                        filtered_definition["name"] = f"Metric {i+1}"
                    if "description" not in filtered_definition:
                        filtered_definition["description"] = definition.get("text", str(definition))
                    definitions.append(MetricDefinition(**filtered_definition))
                else:
                    # If it's a string, use it as the description
                    definitions.append(
                        MetricDefinition(
                            id=f"metric_{i}",
                            name=f"Metric {i+1}",
                            description=str(definition),
                        )
                    )

            # Filter data for DetailedMetricsResponse fields
            filtered_data = self._filter_response_for_dataclass(data, DetailedMetricsResponse)
            filtered_data["definitions"] = definitions
            if "metrics_id" not in filtered_data:
                filtered_data["metrics_id"] = metrics_id
            if "metric_count" not in filtered_data:
                filtered_data["metric_count"] = data.get("num_metrics", len(definitions))
                
            return DetailedMetricsResponse(**filtered_data)
        else:
            response.raise_for_status()
