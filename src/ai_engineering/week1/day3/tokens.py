import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key not found")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"
role = "user"

prompt1 = "Hi"
prompt2 = "Explain python in detail"
prompt3 = "Write a 500 word essay on indian politics"

prompts = [prompt1, prompt2, prompt3]

for prompt in prompts:
    message = {
        "role": role,
        "content": prompt
    }
    messages = [message]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        max_completion_tokens=100
    )
    usage = response.usage
    print(f"Prompt: {prompt} -> Prompt tokens: {usage.prompt_tokens} -> Response tokens: {usage.completion_tokens} -> toal_tokens: {usage.prompt_tokens + usage.completion_tokens} -> Finish Reason: {response.choices[0].finish_reason}")
