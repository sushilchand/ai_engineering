from typing import Optional

from groq import Groq

from ai_engineering.constants import GROQ_API_KEY, LLM_MODEL_NAME


class RAG:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def run_llm(self, question: str) -> str:
        context = self.retrieval(question=question)
        sys_prompt = f"Answer only in one line. Only answer based on context and do not hallucinate. Context: {context}"

        sys_msg = {"role": "system", "content": sys_prompt}

        user_msg = {"role": "user", "content": question}

        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME, messages=[sys_msg, user_msg]
        )
        return response.choices[0].message.content

    def retrieval(self, question: str) -> Optional[str]:
        context = {
            "who": "Sushil is a software engineer",
            "age": "Sushil is 31 years old",
            "occupation": "Sushil is a Senior software engineer",
            "experience": "Sushil has 9 years of experience developing backend systems",
        }
        if "who" in question:
            return context["who"]
        elif "age" in question:
            return context["age"]
        elif "occupation" in question:
            return context["occupation"]
        elif "experience" in question:
            return context["experience"]
        else:
            None


def main():
    rag_obj = RAG()
    # question = "Whos is Sushil ?"
    # question = "What is age"
    question = "Which occupation"
    print(rag_obj.run_llm(question=question.lower()))


if __name__ == "__main__":
    main()
