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
prompt = "Do you know Sushil Chand"
role = "user"
message = {"role": role, "content": prompt}
messages = [message]
models = client.models.list()
# for model in models.data:
#     print(model.id)
response = client.chat.completions.create(model=model, messages=messages)
print(response.choices[0].message.content)
