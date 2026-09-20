import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key not found")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-120b"
prompt = "I love you"
role = "user"
sys_message = {
    "role": "system",
    "content": "You are my wife."
}
message = {
    "role": role,
    "content": prompt
}
messages = [sys_message, message]
models = client.models.list()

response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0,
)
print(response.choices[0].message.content)