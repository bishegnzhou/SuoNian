# Gemini 3 Pro: The Complete Developer's Guide

Gemini 3 Pro is Google's most advanced AI model, featuring state-of-the-art reasoning capabilities. This guide covers everything you need to build agents, run the model, and track costs using the `google-genai` SDK.

## 1. Setup

To use Gemini 3 Pro, you must use the `google-genai` SDK (version 1.51.0 or higher).

```bash
pip install -U google-genai
```

Get your API key from [Google AI Studio](https://aistudio.google.com/).

## 2. Running the Model

Gemini 3 Pro introduces a new parameter: `thinking_level`. This controls the depth of the model's internal reasoning process.

*   **`high`** (Default): Maximum reasoning depth. Best for complex tasks, coding, and math. Higher latency and cost.
*   **`low`**: Faster, lower cost. Good for simple instruction following and chat.

> [!IMPORTANT]
> You cannot disable "thinking" completely for Gemini 3 Pro.

### Basic Example

```python
from google import genai
from google.genai import types
import os

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents="Explain the concept of quantum entanglement to a 5-year-old.",
    config=types.GenerateContentConfig(
        thinking_level="HIGH", # Options: "LOW", "HIGH" (default)
    )
)

print(response.text)
```

## 3. Building an Agent (Tool Use)

Gemini 3 Pro supports advanced tool use (function calling). You define Python functions, pass them to the model, and the model decides when to call them.

### Step-by-Step Agent Example

```python
from google import genai
from google.genai import types
import os

# 1. Define the tools
def get_weather(location: str):
    """Get the current weather for a given location."""
    # In a real app, call a weather API here
    return {"location": location, "temperature": "72", "condition": "Sunny"}

def get_stock_price(ticker: str):
    """Get the current stock price for a given ticker symbol."""
    # In a real app, call a stock API here
    return {"ticker": ticker, "price": "150.25", "currency": "USD"}

# 2. Initialize Client
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# 3. Create the tool configuration
tools = [get_weather, get_stock_price]

# 4. Run the model with tools
response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents="What's the weather in New York and how is Google's stock doing?",
    config=types.GenerateContentConfig(
        tools=tools,
        thinking_level="HIGH" 
    )
)

# 5. Handle the response (Automatic function calling is handled by the SDK in many cases, 
# but here is how you inspect the tool calls if you need to execute them manually or debug)
for part in response.candidates[0].content.parts:
    if part.function_call:
        print(f"Model requested tool: {part.function_call.name}")
        print(f"Arguments: {part.function_call.args}")
        
        # Execute the tool (simplified logic)
        tool_name = part.function_call.name
        tool_args = part.function_call.args
        
        if tool_name == "get_weather":
            result = get_weather(**tool_args)
        elif tool_name == "get_stock_price":
            result = get_stock_price(**tool_args)
            
        print(f"Tool Result: {result}")
```

> [!TIP]
> For a fully autonomous agent, you would feed the tool results back into the model in a loop until the model generates a final text response.

## 4. Cost Tracking & Pricing

Gemini 3 Pro pricing is token-based. To track costs accurately, you must inspect the `usage_metadata` in the response.

### Pricing (Preview Rates)
*   **Input**: ~$2.00 / 1 million tokens (for prompts < 200k tokens)
*   **Output**: ~$12.00 / 1 million tokens (for prompts < 200k tokens)

*Note: Prices increase for context windows > 200k tokens. Always check the [official pricing page](https://ai.google.dev/pricing) for the latest numbers.*

### Tracking Tokens Programmatically

The response object contains a detailed breakdown of token usage, including the new `thoughts_token_count`.

```python
# ... after generating response ...

usage = response.usage_metadata

print(f"Input Tokens: {usage.prompt_token_count}")
print(f"Output Tokens (Candidates): {usage.candidates_token_count}")
print(f"Thinking Tokens: {usage.thoughts_token_count}") 
print(f"Total Tokens: {usage.total_token_count}")

# Simple Cost Calculator (Estimation)
input_cost = (usage.prompt_token_count / 1_000_000) * 2.00
output_cost = (usage.total_token_count - usage.prompt_token_count) / 1_000_000 * 12.00 # Note: Thinking tokens are billed as output
total_cost = input_cost + output_cost

print(f"Estimated Cost: ${total_cost:.6f}")
```

> [!WARNING]
> **Thinking Tokens are Billed as Output**: The `thoughts_token_count` is part of the generated output and is billed at the output token rate. High thinking levels will significantly increase your output token usage and cost.

## 5. Summary of Key Differences

| Feature | Gemini 1.5 Pro | Gemini 3 Pro |
| :--- | :--- | :--- |
| **Reasoning** | Standard | **Advanced (Thinking Process)** |
| **Thinking Control** | N/A | `thinking_level="LOW" | "HIGH"` |
| **Token Metadata** | Standard | Includes `thoughts_token_count` |
| **SDK Requirement** | Older versions OK | Requires `google-genai >= 1.51.0` |
