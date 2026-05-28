import os
import time
from openai import OpenAI


def call_openai(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API and return the response text + latency.
    """

    client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    latency = time.perf_counter() - start_time

    response_text = response.choices[0].message.content

    return response_text, latency

# TEST NHANH
response, latency = call_openai(
    prompt="Hello, introduce yourself."
)

print(response)
print(f"Latency: {latency:.2f}s")