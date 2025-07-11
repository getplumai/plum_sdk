import requests
from typing import List, Optional
from .models import (
    TrainingExample,
    UploadResponse,
    MetricsQuestions,
    MetricsResponse,
    EvaluationResponse,
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

    def evaluate(
        self,
        data_id: str,
        metrics_id: str,
        latest_n_pairs: Optional[int] = None,
        pair_label: Optional[str] = None,
        is_synthetic: bool = False
    ) -> EvaluationResponse:
        """
        Evaluate a dataset using specified metrics.

        Args:
            data_id: The ID of the dataset to evaluate
            metrics_id: The ID of the metrics to use for evaluation
            latest_n_pairs: Maximum number of latest pairs to include (defaults to 150 if not provided)
            pair_label: Filter pairs by label (optional)
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
            or pair_label is not None
            or last_n_seconds is not None
        ):
            pair_query = {}
            if latest_n_pairs is not None:
                pair_query["latest_n_pairs"] = latest_n_pairs
            if pair_label is not None:
                pair_query["pair_label"] = pair_label
            payload["pair_query"] = pair_query

        response = requests.post(url, json=payload, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            return EvaluationResponse(**data)
        else:
            response.raise_for_status()

    def augment(
        self,
        seed_data_id: Optional[str] = None,
        multiple: int = 1,
        eval_results_id: Optional[str] = None,
        latest_n_pairs: Optional[int] = None,
        pair_label: Optional[str] = None,
        target_metric: Optional[str] = None,
    ) -> dict:
        """
        Augment seed data to generate synthetic data.

        Args:
            seed_data_id: ID of seed dataset to augment (will use latest if not provided)
            multiple: Number of synthetic examples to generate per seed example (max 50)
            eval_results_id: ID of evaluation results to use for target metric (will use latest if not provided)
            latest_n_pairs: Maximum number of latest pairs to include (defaults to 150 if not provided)
            pair_label: Filter pairs by label (optional)
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
        if latest_n_pairs is not None or pair_label is not None:
            pair_query = {}
            if latest_n_pairs is not None:
                pair_query["latest_n_pairs"] = latest_n_pairs
            if pair_label is not None:
                pair_query["pair_label"] = pair_label
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
            metadata = None
            if "metadata" in pair_data:
                metadata = IOPairMeta(
                    created_at=pair_data["metadata"].get("created_at"),
                    labels=pair_data["metadata"].get("labels", []),
                )

            pairs.append(
                IOPair(
                    id=pair_data["id"],
                    input=pair_data["input"],
                    output=pair_data["output"],
                    metadata=metadata,
                    input_media=pair_data.get("input_media"),
                    use_media_mime_type=pair_data.get("use_media_mime_type"),
                    human_critique=pair_data.get("human_critique"),
                    target_metric=pair_data.get("target_metric"),
                )
            )

        return Dataset(
            id=data["id"],
            data=pairs,
            system_prompt=data.get("system_prompt"),
            created_at=data.get("created_at"),
        )

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
                # Convert definitions list to MetricDefinition objects
                definitions = []
                for i, definition in enumerate(metric_data.get("definitions", [])):
                    # Handle different formats of definition data
                    if isinstance(definition, dict):
                        definitions.append(
                            MetricDefinition(
                                id=definition.get("id", f"metric_{i}"),
                                name=definition.get("name", f"Metric {i+1}"),
                                description=definition.get(
                                    "description",
                                    definition.get("text", str(definition)),
                                ),
                            )
                        )
                    else:
                        # If it's a string, use it as the description
                        definitions.append(
                            MetricDefinition(
                                id=f"metric_{i}",
                                name=f"Metric {i+1}",
                                description=str(definition),
                            )
                        )

                metrics_dict[metric_id] = DetailedMetricsResponse(
                    metrics_id=metric_data.get("metrics_id", metric_id),
                    definitions=definitions,
                    system_prompt=metric_data.get("system_prompt"),
                    metric_count=metric_data.get("metric_count", len(definitions)),
                    created_at=metric_data.get("created_at"),
                )

            return MetricsListResponse(
                metrics=metrics_dict,
                total_count=data.get("total_count", len(metrics_dict)),
            )
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
                    definitions.append(
                        MetricDefinition(
                            id=definition.get("id", f"metric_{i}"),
                            name=definition.get("name", f"Metric {i+1}"),
                            description=definition.get(
                                "description", definition.get("text", str(definition))
                            ),
                        )
                    )
                else:
                    # If it's a string, use it as the description
                    definitions.append(
                        MetricDefinition(
                            id=f"metric_{i}",
                            name=f"Metric {i+1}",
                            description=str(definition),
                        )
                    )

            return DetailedMetricsResponse(
                metrics_id=data.get("metrics_id", metrics_id),
                definitions=definitions,
                system_prompt=data.get("system_prompt"),
                metric_count=data.get("num_metrics", len(definitions)),
                created_at=data.get("created_at"),
            )
        else:
            response.raise_for_status()
