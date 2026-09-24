from groq import Groq

from ai_engineering.constants import GROQ_API_KEY, LLM_MODEL_NAME


def main():
    client = Groq(api_key=GROQ_API_KEY)

    sys_prompt = "You are an expert teacher of Machine learning"
    sys_msg = {"role": "system", "content": sys_prompt}

    prompt = "Explain how LLM works and also give 5 examples"
    user_msg = {"role": "user", "content": prompt}

    # response = client.chat.completions.create(
    #     model=LLM_MODEL_NAME,
    #     messages=[sys_msg, user_msg]
    # )
    # print(response.choices[0].message.content)
    streaming_data = client.chat.completions.create(
        model=LLM_MODEL_NAME, messages=[sys_msg, user_msg], stream=True
    )
    for chunk in streaming_data:
        data = chunk.choices[0].delta.content
        if data:
            print(data, end="", flush=True)


if __name__ == "__main__":
    main()
