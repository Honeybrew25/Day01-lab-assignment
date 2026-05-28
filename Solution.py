import os
import time
from openai import OpenAI

COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-4o": 0.01,
    "gpt-4o-mini": 0.0006,
}

def call_openai(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:

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


def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:

    return call_openai(
        prompt=prompt,
        model="gpt-4o-mini",
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
response, latency = call_openai_mini(
    prompt="What is Python?",
    temperature=0.7,
    max_tokens=50
)

print("Response:")
print(response)
print(f"\nLatency: {latency:.2f} seconds")


def compare_models(prompt: str) -> dict:
    # GPT-4o
    gpt4o_response, gpt4o_latency = call_openai(prompt)

    # GPT-4o-mini
    mini_response, mini_latency = call_openai_mini(prompt)

    # Estimate output tokens
    estimated_tokens = len(gpt4o_response.split()) / 0.75

    # Estimate cost
    gpt4o_cost_estimate = (
        estimated_tokens / 1000
    ) * COST_PER_1K_OUTPUT_TOKENS["gpt-4o"]

    return {
        "gpt4o_response": gpt4o_response,
        "mini_response": mini_response,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": gpt4o_cost_estimate,
    }

result = compare_models("Explain machine learning in one sentence.")

print(result)

def streaming_chatbot() -> None:
    history = []

    print("Streaming chatbot started.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        # Add user message to history
        history.append({
            "role": "user",
            "content": user_input
        })

        print("Assistant: ", end="", flush=True)

        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history,
            stream=True
        )

        assistant_reply = ""

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""

            print(delta, end="", flush=True)

            assistant_reply += delta

        print("\n")

        # Save assistant response
        history.append({
            "role": "assistant",
            "content": assistant_reply
        })

        # Keep only last 3 turns
        history = history[-6:]