#!/usr/bin/env python3
"""
Test script to verify that the new dataclasses are properly exported
"""


def test_imports():
    """Test that all new dataclasses can be imported from the top level"""
    try:
        from plum_sdk import IOPair, IOPairMeta, Dataset, PlumClient

        print("✅ Successfully imported IOPair, IOPairMeta, Dataset, PlumClient")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False

    try:
        # Test that we can create instances
        metadata = IOPairMeta(created_at="2023-01-01T00:00:00Z", labels=["test"])
        pair = IOPair(
            id="test_pair", input="test input", output="test output", metadata=metadata
        )
        dataset = Dataset(id="test_dataset", data=[pair], system_prompt="test prompt")
        client = PlumClient("test_key")

        print("✅ Successfully created instances of all new dataclasses")
        print(f"   - IOPairMeta: {metadata}")
        print(f"   - IOPair: {pair.id}")
        print(f"   - Dataset: {dataset.id} with {len(dataset.data)} pairs")
        print(f"   - PlumClient: {client.base_url}")

    except Exception as e:
        print(f"❌ Failed to create instances: {e}")
        return False

    return True


def test_dataclass_features():
    """Test that dataclasses have expected features"""
    try:
        from plum_sdk import IOPair, IOPairMeta, Dataset

        # Test equality
        meta1 = IOPairMeta(created_at="2023-01-01T00:00:00Z", labels=["test"])
        meta2 = IOPairMeta(created_at="2023-01-01T00:00:00Z", labels=["test"])

        assert meta1 == meta2, "IOPairMeta equality should work"
        print("✅ IOPairMeta equality works")

        # Test string representation
        pair = IOPair(id="test", input="test", output="test", metadata=meta1)
        pair_str = str(pair)
        assert "IOPair" in pair_str, "String representation should contain class name"
        print("✅ IOPair string representation works")

        # Test optional fields
        minimal_pair = IOPair(id="minimal", input="input", output="output")
        assert minimal_pair.metadata is None, "Optional fields should default to None"
        print("✅ Optional fields work correctly")

        return True

    except Exception as e:
        print(f"❌ Dataclass features test failed: {e}")
        return False


def main():
    print("Testing Plum SDK new dataclasses...")
    print("=" * 50)

    success = True
    success &= test_imports()
    success &= test_dataclass_features()

    print("=" * 50)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")

    return success


if __name__ == "__main__":
    main()
