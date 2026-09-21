import os

from dotenv import load_dotenv
from groq import Groq

from ai_engineering.week1.day4.schema import UserSchema

load_dotenv()


def main():
    my_api_key = os.getenv("GROQ_API_KEY")
    if not my_api_key:
        raise ValueError("API key not found")

    client = Groq(api_key=my_api_key)
    model = "openai/gpt-oss-120b"

    schema = UserSchema.model_json_schema()
    response_format = {"type": "json_object"}

    sys_prompt = f"Extract information strictly based on this json schema {schema}"
    sys_message = {"role": "system", "content": sys_prompt}

    prompt = "Give me details of Virat Kohli, his wife, parents and children"
    message = {"role": "user", "content": prompt}

    messages = [sys_message, message]

    response = client.chat.completions.create(
        model=model, messages=messages, response_format=response_format
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
