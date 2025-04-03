import unittest
from plum_sdk import TrainingExample


class TestTrainingExample(unittest.TestCase):
    def test_training_example_creation(self):
        example = TrainingExample(input="test input", output="test output")
        self.assertEqual(example.input, "test input")
        self.assertEqual(example.output, "test output")
