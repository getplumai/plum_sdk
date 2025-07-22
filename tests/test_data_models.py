import unittest
from plum_sdk.models import (
    IOPair,
    IOPairMeta,
    Dataset,
    TrainingExample,
    UploadResponse,
    MetricDefinition,
    DetailedMetricsResponse,
    MetricsListResponse,
)


class TestDataModels(unittest.TestCase):
    """Test the dataclass models for type safety and proper initialization"""

    def test_iopair_meta_creation(self):
        """Test IOPairMeta creation with various parameters"""
        # Test with no parameters
        meta = IOPairMeta()
        self.assertIsNone(meta.created_at)
        self.assertIsNone(meta.labels)

        # Test with created_at only
        meta_with_date = IOPairMeta(created_at="2023-01-01T00:00:00Z")
        self.assertEqual(meta_with_date.created_at, "2023-01-01T00:00:00Z")
        self.assertIsNone(meta_with_date.labels)

        # Test with labels only
        meta_with_labels = IOPairMeta(labels=["test", "example"])
        self.assertIsNone(meta_with_labels.created_at)
        self.assertEqual(meta_with_labels.labels, ["test", "example"])

        # Test with both parameters
        meta_full = IOPairMeta(
            created_at="2023-01-01T00:00:00Z", labels=["test", "example", "complete"]
        )
        self.assertEqual(meta_full.created_at, "2023-01-01T00:00:00Z")
        self.assertEqual(meta_full.labels, ["test", "example", "complete"])

    def test_iopair_creation(self):
        """Test IOPair creation with various parameters"""
        # Test minimal required parameters
        pair = IOPair(id="test_pair", input="test input", output="test output")
        self.assertEqual(pair.id, "test_pair")
        self.assertEqual(pair.input, "test input")
        self.assertEqual(pair.output, "test output")
        self.assertIsNone(pair.metadata)
        self.assertIsNone(pair.input_media)
        self.assertIsNone(pair.use_media_mime_type)
        self.assertIsNone(pair.human_critique)
        self.assertIsNone(pair.target_metric)

        # Test with metadata
        metadata = IOPairMeta(created_at="2023-01-01T00:00:00Z", labels=["test"])
        pair_with_meta = IOPair(
            id="test_pair_meta",
            input="test input",
            output="test output",
            metadata=metadata,
        )
        self.assertEqual(pair_with_meta.metadata.created_at, "2023-01-01T00:00:00Z")
        self.assertEqual(pair_with_meta.metadata.labels, ["test"])

        # Test with all parameters
        pair_full = IOPair(
            id="full_pair",
            input="full input",
            output="full output",
            metadata=metadata,
            input_media=b"fake_media_bytes",
            use_media_mime_type="image/jpeg",
            human_critique="Good response",
            target_metric="accuracy",
        )
        self.assertEqual(pair_full.input_media, b"fake_media_bytes")
        self.assertEqual(pair_full.use_media_mime_type, "image/jpeg")
        self.assertEqual(pair_full.human_critique, "Good response")
        self.assertEqual(pair_full.target_metric, "accuracy")

    def test_dataset_creation(self):
        """Test Dataset creation with various parameters"""
        # Test minimal required parameters
        dataset = Dataset(id="test_dataset", data=[])
        self.assertEqual(dataset.id, "test_dataset")
        self.assertEqual(dataset.data, [])
        self.assertIsNone(dataset.system_prompt)
        self.assertIsNone(dataset.created_at)

        # Test with pairs
        pair1 = IOPair(id="pair1", input="input1", output="output1")
        pair2 = IOPair(id="pair2", input="input2", output="output2")
        dataset_with_pairs = Dataset(id="dataset_with_pairs", data=[pair1, pair2])
        self.assertEqual(len(dataset_with_pairs.data), 2)
        self.assertEqual(dataset_with_pairs.data[0].id, "pair1")
        self.assertEqual(dataset_with_pairs.data[1].id, "pair2")

        # Test with all parameters
        dataset_full = Dataset(
            id="full_dataset",
            data=[pair1],
            system_prompt="You are a helpful assistant",
            created_at="2023-01-01T00:00:00Z",
        )
        self.assertEqual(dataset_full.system_prompt, "You are a helpful assistant")
        self.assertEqual(dataset_full.created_at, "2023-01-01T00:00:00Z")

    def test_dataset_with_complex_pairs(self):
        """Test Dataset with complex IOPair objects"""
        metadata1 = IOPairMeta(
            created_at="2023-01-01T00:00:00Z", labels=["complex", "test"]
        )
        metadata2 = IOPairMeta(
            created_at="2023-01-02T00:00:00Z", labels=["media", "image"]
        )

        pair1 = IOPair(
            id="complex_pair1",
            input="Analyze this text",
            output="This is a text analysis",
            metadata=metadata1,
            human_critique="Good analysis",
            target_metric="relevance",
        )

        pair2 = IOPair(
            id="complex_pair2",
            input="Describe this image",
            output="This is an image description",
            metadata=metadata2,
            input_media=b"fake_image_data",
            use_media_mime_type="image/png",
            human_critique="Accurate description",
            target_metric="accuracy",
        )

        dataset = Dataset(
            id="complex_dataset",
            data=[pair1, pair2],
            system_prompt="You are an expert analyst",
            created_at="2023-01-01T00:00:00Z",
        )

        # Verify the dataset structure
        self.assertEqual(len(dataset.data), 2)

        # Verify first pair
        self.assertEqual(dataset.data[0].id, "complex_pair1")
        self.assertEqual(dataset.data[0].metadata.labels, ["complex", "test"])
        self.assertEqual(dataset.data[0].human_critique, "Good analysis")
        self.assertIsNone(dataset.data[0].input_media)

        # Verify second pair
        self.assertEqual(dataset.data[1].id, "complex_pair2")
        self.assertEqual(dataset.data[1].metadata.labels, ["media", "image"])
        self.assertEqual(dataset.data[1].input_media, b"fake_image_data")
        self.assertEqual(dataset.data[1].use_media_mime_type, "image/png")

    def test_dataclass_equality(self):
        """Test that dataclass instances can be compared for equality"""
        meta1 = IOPairMeta(created_at="2023-01-01T00:00:00Z", labels=["test"])
        meta2 = IOPairMeta(created_at="2023-01-01T00:00:00Z", labels=["test"])
        meta3 = IOPairMeta(created_at="2023-01-02T00:00:00Z", labels=["test"])

        self.assertEqual(meta1, meta2)
        self.assertNotEqual(meta1, meta3)

        pair1 = IOPair(id="test", input="input", output="output", metadata=meta1)
        pair2 = IOPair(id="test", input="input", output="output", metadata=meta2)
        pair3 = IOPair(id="test", input="input", output="output", metadata=meta3)

        self.assertEqual(pair1, pair2)
        self.assertNotEqual(pair1, pair3)

    def test_dataclass_string_representation(self):
        """Test string representation of dataclasses"""
        meta = IOPairMeta(created_at="2023-01-01T00:00:00Z", labels=["test"])
        pair = IOPair(
            id="test_pair", input="test input", output="test output", metadata=meta
        )
        dataset = Dataset(id="test_dataset", data=[pair], system_prompt="test prompt")

        # Test that string representations are meaningful
        meta_str = str(meta)
        self.assertIn("IOPairMeta", meta_str)
        self.assertIn("2023-01-01T00:00:00Z", meta_str)

        pair_str = str(pair)
        self.assertIn("IOPair", pair_str)
        self.assertIn("test_pair", pair_str)

        dataset_str = str(dataset)
        self.assertIn("Dataset", dataset_str)
        self.assertIn("test_dataset", dataset_str)

    def test_dataclass_type_hints(self):
        """Test that dataclasses maintain proper type information"""
        from typing import get_type_hints

        # Test IOPairMeta type hints
        meta_hints = get_type_hints(IOPairMeta)
        self.assertIn("created_at", meta_hints)
        self.assertIn("labels", meta_hints)

        # Test IOPair type hints
        pair_hints = get_type_hints(IOPair)
        self.assertIn("id", pair_hints)
        self.assertIn("input", pair_hints)
        self.assertIn("output", pair_hints)
        self.assertIn("metadata", pair_hints)

        # Test Dataset type hints
        dataset_hints = get_type_hints(Dataset)
        self.assertIn("id", dataset_hints)
        self.assertIn("data", dataset_hints)

    def test_empty_labels_handling(self):
        """Test handling of empty labels lists"""
        meta_empty = IOPairMeta(labels=[])
        meta_none = IOPairMeta(labels=None)

        self.assertEqual(meta_empty.labels, [])
        self.assertIsNone(meta_none.labels)

        pair_empty = IOPair(id="test", input="test", output="test", metadata=meta_empty)
        pair_none = IOPair(id="test", input="test", output="test", metadata=meta_none)

        self.assertEqual(pair_empty.metadata.labels, [])
        self.assertIsNone(pair_none.metadata.labels)

    def test_binary_media_handling(self):
        """Test handling of binary media data"""
        # Test with actual binary data
        binary_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"

        pair = IOPair(
            id="media_pair",
            input="Analyze this image",
            output="Image analysis",
            input_media=binary_data,
            use_media_mime_type="image/png",
        )

        self.assertEqual(pair.input_media, binary_data)
        self.assertEqual(pair.use_media_mime_type, "image/png")

        # Test with empty bytes
        pair_empty = IOPair(
            id="empty_media",
            input="No media",
            output="Response",
            input_media=b"",
            use_media_mime_type="application/octet-stream",
        )

        self.assertEqual(pair_empty.input_media, b"")

    def test_metric_definition_creation(self):
        """Test MetricDefinition creation with various parameters"""
        # Test minimal required parameters
        metric = MetricDefinition(
            id="test_metric",
            name="Test Metric",
            description="A test metric for validation",
        )
        self.assertEqual(metric.id, "test_metric")
        self.assertEqual(metric.name, "Test Metric")
        self.assertEqual(metric.description, "A test metric for validation")

        # Test with empty description
        metric_empty = MetricDefinition(
            id="empty_metric", name="Empty Metric", description=""
        )
        self.assertEqual(metric_empty.description, "")

        # Test with long description
        long_desc = "This is a very long description that tests how the metric definition handles longer text content that might be used in real-world scenarios."
        metric_long = MetricDefinition(
            id="long_metric", name="Long Description Metric", description=long_desc
        )
        self.assertEqual(metric_long.description, long_desc)

    def test_detailed_metrics_response_creation(self):
        """Test DetailedMetricsResponse creation with various parameters"""
        # Test minimal required parameters
        detailed_metrics = DetailedMetricsResponse(
            metrics_id="test_metrics_id", definitions=[]
        )
        self.assertEqual(detailed_metrics.metrics_id, "test_metrics_id")
        self.assertEqual(detailed_metrics.definitions, [])
        self.assertIsNone(detailed_metrics.system_prompt)
        self.assertEqual(detailed_metrics.metric_count, 0)
        self.assertIsNone(detailed_metrics.created_at)

        # Test with single metric definition
        metric_def = MetricDefinition(
            id="accuracy", name="Accuracy", description="Measures response accuracy"
        )
        detailed_with_def = DetailedMetricsResponse(
            metrics_id="accuracy_metrics", definitions=[metric_def], metric_count=1
        )
        self.assertEqual(len(detailed_with_def.definitions), 1)
        self.assertEqual(detailed_with_def.definitions[0].id, "accuracy")
        self.assertEqual(detailed_with_def.metric_count, 1)

        # Test with multiple metric definitions
        metric_def2 = MetricDefinition(
            id="relevance", name="Relevance", description="Measures response relevance"
        )
        detailed_multiple = DetailedMetricsResponse(
            metrics_id="multi_metrics",
            definitions=[metric_def, metric_def2],
            system_prompt="You are a helpful assistant",
            metric_count=2,
            created_at="2023-01-01T00:00:00Z",
        )
        self.assertEqual(len(detailed_multiple.definitions), 2)
        self.assertEqual(detailed_multiple.system_prompt, "You are a helpful assistant")
        self.assertEqual(detailed_multiple.metric_count, 2)
        self.assertEqual(detailed_multiple.created_at, "2023-01-01T00:00:00Z")

    def test_metrics_list_response_creation(self):
        """Test MetricsListResponse creation with various parameters"""
        # Test empty metrics list
        empty_list = MetricsListResponse(metrics={}, total_count=0)
        self.assertEqual(empty_list.metrics, {})
        self.assertEqual(empty_list.total_count, 0)

        # Test with single metrics entry
        metric_def = MetricDefinition(
            id="test_metric", name="Test Metric", description="Test description"
        )
        detailed_metrics = DetailedMetricsResponse(
            metrics_id="test_id", definitions=[metric_def], metric_count=1
        )
        single_list = MetricsListResponse(
            metrics={"test_id": detailed_metrics}, total_count=1
        )
        self.assertEqual(single_list.total_count, 1)
        self.assertIn("test_id", single_list.metrics)
        self.assertEqual(single_list.metrics["test_id"].metrics_id, "test_id")

        # Test with multiple metrics entries
        metric_def2 = MetricDefinition(
            id="accuracy", name="Accuracy", description="Accuracy metric"
        )
        detailed_metrics2 = DetailedMetricsResponse(
            metrics_id="accuracy_id", definitions=[metric_def2], metric_count=1
        )
        multi_list = MetricsListResponse(
            metrics={"test_id": detailed_metrics, "accuracy_id": detailed_metrics2},
            total_count=2,
        )
        self.assertEqual(multi_list.total_count, 2)
        self.assertEqual(len(multi_list.metrics), 2)
        self.assertIn("test_id", multi_list.metrics)
        self.assertIn("accuracy_id", multi_list.metrics)

    def test_metrics_dataclass_equality(self):
        """Test equality comparison for metrics dataclasses"""
        # Test MetricDefinition equality
        metric1 = MetricDefinition(id="test", name="Test", description="Description")
        metric2 = MetricDefinition(id="test", name="Test", description="Description")
        metric3 = MetricDefinition(
            id="test", name="Test", description="Different description"
        )

        self.assertEqual(metric1, metric2)
        self.assertNotEqual(metric1, metric3)

        # Test DetailedMetricsResponse equality
        detailed1 = DetailedMetricsResponse(
            metrics_id="test", definitions=[metric1], metric_count=1
        )
        detailed2 = DetailedMetricsResponse(
            metrics_id="test", definitions=[metric2], metric_count=1
        )
        detailed3 = DetailedMetricsResponse(
            metrics_id="different", definitions=[metric1], metric_count=1
        )

        self.assertEqual(detailed1, detailed2)
        self.assertNotEqual(detailed1, detailed3)

    def test_metrics_string_representation(self):
        """Test string representation of metrics dataclasses"""
        metric = MetricDefinition(
            id="test_metric", name="Test Metric", description="Test description"
        )

        detailed = DetailedMetricsResponse(
            metrics_id="test_id", definitions=[metric], metric_count=1
        )

        metrics_list = MetricsListResponse(metrics={"test_id": detailed}, total_count=1)

        # Test string representations contain key information
        metric_str = str(metric)
        self.assertIn("MetricDefinition", metric_str)
        self.assertIn("test_metric", metric_str)

        detailed_str = str(detailed)
        self.assertIn("DetailedMetricsResponse", detailed_str)
        self.assertIn("test_id", detailed_str)

        list_str = str(metrics_list)
        self.assertIn("MetricsListResponse", list_str)

    def test_metrics_type_hints(self):
        """Test that metrics dataclasses maintain proper type information"""
        from typing import get_type_hints

        # Test MetricDefinition type hints
        metric_hints = get_type_hints(MetricDefinition)
        self.assertIn("id", metric_hints)
        self.assertIn("name", metric_hints)
        self.assertIn("description", metric_hints)

        # Test DetailedMetricsResponse type hints
        detailed_hints = get_type_hints(DetailedMetricsResponse)
        self.assertIn("metrics_id", detailed_hints)
        self.assertIn("definitions", detailed_hints)
        self.assertIn("system_prompt", detailed_hints)
        self.assertIn("metric_count", detailed_hints)
        self.assertIn("created_at", detailed_hints)

        # Test MetricsListResponse type hints
        list_hints = get_type_hints(MetricsListResponse)
        self.assertIn("metrics", list_hints)
        self.assertIn("total_count", list_hints)

    def test_complex_metrics_scenario(self):
        """Test a complex scenario with multiple metrics and detailed responses"""
        # Create multiple metric definitions
        accuracy_metric = MetricDefinition(
            id="accuracy",
            name="Accuracy",
            description="Measures how accurate the response is",
        )

        relevance_metric = MetricDefinition(
            id="relevance",
            name="Relevance",
            description="Measures how relevant the response is to the input",
        )

        clarity_metric = MetricDefinition(
            id="clarity",
            name="Clarity",
            description="Measures how clear and understandable the response is",
        )

        # Create detailed metrics responses
        qa_metrics = DetailedMetricsResponse(
            metrics_id="qa_metrics",
            definitions=[accuracy_metric, relevance_metric],
            system_prompt="You are a helpful Q&A assistant",
            metric_count=2,
            created_at="2023-01-01T00:00:00Z",
        )

        writing_metrics = DetailedMetricsResponse(
            metrics_id="writing_metrics",
            definitions=[clarity_metric, relevance_metric],
            system_prompt="You are a writing assistant",
            metric_count=2,
            created_at="2023-01-02T00:00:00Z",
        )

        # Create metrics list response
        all_metrics = MetricsListResponse(
            metrics={"qa_metrics": qa_metrics, "writing_metrics": writing_metrics},
            total_count=2,
        )

        # Verify the complex structure
        self.assertEqual(all_metrics.total_count, 2)
        self.assertEqual(len(all_metrics.metrics), 2)

        # Verify QA metrics
        qa = all_metrics.metrics["qa_metrics"]
        self.assertEqual(qa.metrics_id, "qa_metrics")
        self.assertEqual(len(qa.definitions), 2)
        self.assertEqual(qa.definitions[0].id, "accuracy")
        self.assertEqual(qa.definitions[1].id, "relevance")
        self.assertEqual(qa.metric_count, 2)

        # Verify writing metrics
        writing = all_metrics.metrics["writing_metrics"]
        self.assertEqual(writing.metrics_id, "writing_metrics")
        self.assertEqual(len(writing.definitions), 2)
        self.assertEqual(writing.definitions[0].id, "clarity")
        self.assertEqual(writing.definitions[1].id, "relevance")
        self.assertEqual(writing.metric_count, 2)

    def test_metrics_edge_cases(self):
        """Test edge cases for metrics dataclasses"""
        # Test with special characters in descriptions
        special_metric = MetricDefinition(
            id="special_chars",
            name="Special Characters: !@#$%^&*()",
            description="Description with special chars: <>&\"'",
        )
        self.assertIn("!@#$%^&*()", special_metric.name)
        self.assertIn("<>&\"'", special_metric.description)

        # Test with very long metric ID
        long_id = "a" * 100
        long_metric = MetricDefinition(
            id=long_id, name="Long ID Metric", description="Test long ID"
        )
        self.assertEqual(long_metric.id, long_id)

        # Test with zero metric count but non-empty definitions
        inconsistent_metrics = DetailedMetricsResponse(
            metrics_id="inconsistent",
            definitions=[special_metric],
            metric_count=0,  # Intentionally inconsistent
        )
        self.assertEqual(inconsistent_metrics.metric_count, 0)
        self.assertEqual(len(inconsistent_metrics.definitions), 1)

        # Test with negative total count (edge case)
        negative_count = MetricsListResponse(metrics={}, total_count=-1)
        self.assertEqual(negative_count.total_count, -1)


if __name__ == "__main__":
    unittest.main()
