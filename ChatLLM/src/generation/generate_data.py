import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"

def call_ollama(prompt: str, model: str = MODEL) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,  # get the full response at once, not token-by-token
        },
    )
    response.raise_for_status()
    return response.json()["response"]


if __name__ == "__main__":
    test_prompt = "Skriv 3 beskeder fra en chatbruger der taber penge på spil men bagatelliserer det."
    result = call_ollama(test_prompt)
    print(result)