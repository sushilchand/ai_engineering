import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

from ai_engineering.constants import GROQ_API_KEY, LLM_MODEL_NAME


class FullRag:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.doc = [
            "Employees receive 24 days of paid leave per year.",
            "Employees work from the office on Tuesday, Wednesday and Thursday. "
            "Monday and Friday are optional work-from-home days.",
            "Employees receive Rs 3000 per month for gym reimbursement.",
            "Employees can claim Rs 2000 per month for home internet.",
            "Employees have a 90 day notice period.",
        ]

    def get_cosine_similarity(self, vector_a, vector_b):
        return np.dot(vector_a, vector_b) / (
            np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        )

    def run_llm(self, question, context):
        sys_prompt = f"Answer only in one line and do not hallucinate. Answer based on given context only. Context: {context}"

        sys_msg = {"role": "system", "content": sys_prompt}

        user_msg = {"role": "user", "content": question}

        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME, messages=[sys_msg, user_msg]
        )
        return response.choices[0].message.content

    def get_vector_embeding(self) -> list:

        embeded_doc = self.model.encode(self.doc)
        return embeded_doc

    def retrieve_context(self, question: str, embeded_vector: list) -> tuple:
        scores: list[tuple[float, str]] = []
        embeded_question = self.model.encode(question)
        for i, vector in enumerate(embeded_vector):
            score = self.get_cosine_similarity(
                vector_a=embeded_question, vector_b=vector
            )
            scores.append((score, self.doc[i]))

        scores.sort(reverse=True)
        return scores[0]

    def get_answer(self, question: str) -> str:
        embeded_docs = self.get_vector_embeding()
        _, context = self.retrieve_context(
            question=question, embeded_vector=embeded_docs
        )

        print(context)

        return self.run_llm(question=question, context=context)


def main():
    rag_obj = FullRag()
    print(rag_obj.get_answer(question="Coming to office compulsary?"))


if __name__ == "__main__":
    main()
