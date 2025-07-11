from setuptools import setup, find_packages

setup(
    name="plum_sdk",
    version="0.4.0",
    packages=find_packages(),
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
