from setuptools import setup, find_packages

setup(
    name='plum_sdk',
    version='0.0.3',
    packages=find_packages(),
    install_requires=['requests'],
    description='Python SDK for Plum AI',
    author='Plum AI',
    author_email='founders@getplum.ai',
    url='https://github.com/getplumai/plum_sdk',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
