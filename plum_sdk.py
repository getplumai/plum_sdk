import requests
from dataclasses import dataclass
from typing import List

@dataclass
class TrainingExample:
    input: str
    output: str

class PlumSDK:
    def __init__(self, api_key: str, base_url: str = "http://beta.getplum.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url

    def upload_data(self, training_examples: List[TrainingExample], system_prompt: str):
        url = f"{self.base_url}/data/seed"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"{self.api_key}"
        }

        data = [
            {"input": example.input, "output": example.output}
            for example in training_examples
        ]

        payload = {
            "data": data,
            "system_prompt": system_prompt
        }

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()