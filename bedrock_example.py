import boto3
import json
import os

# API key-ul se pune în fișierul .env la AWS_BEARER_TOKEN_BEDROCK

os.environ["AWS_BEARER_TOKEN_BEDROCK"] = "bedrock api hey here ..."


def call_claude(
    prompt: str, model_id: str = "global.anthropic.claude-opus-4-6-v1"
) -> str:
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
    )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]


if __name__ == "__main__":
    reply = call_claude("What is the capital of France?")
    print("Response:", reply)
