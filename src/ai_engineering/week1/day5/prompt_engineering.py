from groq import Groq

from ai_engineering.constants import GROQ_API_KEY, LLM_MODEL_NAME


def main():
    client = Groq(api_key=GROQ_API_KEY)
    # bad_prompt = """
    # This is a user complaint:
    # My Laptop is not working
    # """
    prompt = """
    #ROLE
    You are an customer executive
    #TASK
    Classify the issue in a category
    #CONSTRAINT
    Classify in these cateogires billing, return, technical
    #OUTPUT FORMAT
    The answer should only be in a single word
    #EXAMPLE
    If user says he wants refund then cateogry is return
    #FALLBACK
    If issue doesn't fall in any category then return OTHER
    This is a user complaint:
    Didn't recieve my cashback of laptop purchase
    """
    message = {"role": "user", "content": prompt}

    response = client.chat.completions.create(model=LLM_MODEL_NAME, messages=[message])
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
