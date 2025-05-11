from setuptools import setup, find_packages
import os

# Check if we have a plum_sdk directory with actual code
if not os.path.exists("plum_sdk") and os.path.exists("__init__.py"):
    # We're inside the package directory - use . to include the current directory
    package_dir = {"": "."}
    packages = [""]
else:
    # Regular package structure
    packages = find_packages(include=["plum_sdk", "plum_sdk.*"])
    package_dir = {}

setup(
    name="plum_sdk",
    version="0.1.7",
    packages=packages,
    package_dir=package_dir,
    py_modules=["plum_sdk"] if not packages else [],
    install_requires=["requests"],
    tests_require=["pytest"],
    python_requires=">=3.6",
    description="Python SDK for Plum AI",
    author="Plum AI",
    author_email="founders@getplum.ai",
    url="https://github.com/getplumai/plum_sdk",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    extras_require={
        "dev": ["black"],
    },
)
