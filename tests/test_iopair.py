import unittest
from plum_sdk import IOPair


class TestIOPair(unittest.TestCase):
    def test_iopair_creation(self):
        example = IOPair(input="test input", output="test output")
        self.assertEqual(example.input, "test input")
        self.assertEqual(example.output, "test output")
