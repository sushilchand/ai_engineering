import json
import time

from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadFieldSchema,
    PayloadSchemaType,
    PointStruct,
    ScoredPoint,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from ai_engineering.constants import (
    GROQ_API_KEY,
    LLM_MODEL_NAME,
    QDRANT_API_KEY,
    QDRANT_EMBEDDING_SIZE,
    QDRANT_ENDPOINT,
)

COLLECTION_NAME = "rag_knowledge_base"


class Quadrant:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)
        self.sentence_transformer = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def initialise(self, delete_colleciton: bool = True):
        if delete_colleciton:
            if self.client.collection_exists(collection_name=COLLECTION_NAME):
                self.client.delete_collection(collection_name=COLLECTION_NAME)
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=QDRANT_EMBEDDING_SIZE,
                    distance=Distance.COSINE,
                ),
            )

            self.create_index()

            self.upsert_data()

    def create_index(self):
        self.client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="category",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def upsert_data(self):
        with open("src/ai_engineering/week4/day1/knowledge.json", "r") as fp:
            documents = json.load(fp=fp)

        text_list = [doc["text"] for doc in documents]

        embeded_doc_list = self.sentence_transformer.encode(text_list)

        points: list[PointStruct] = []

        for i, e_doc in enumerate(embeded_doc_list):
            point = PointStruct(id=i + 1, vector=e_doc.tolist(), payload=documents[i])
            points.append(point)

        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Successfully updated {len(points)} to vector db")

    def retrieve(self, question: str, top_k: int = 3) -> list[ScoredPoint]:
        embeded_query = self.sentence_transformer.encode(question)

        result = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=list(embeded_query),
            limit=top_k,
            with_payload=True,
        ).points
        return result


class RAG:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def ask_llm(self, question: str, context: str):
        prompt = f"""
            Answer only based on the context provided

            Question: {question}

            Context: {context}

            If answer is not persent in the context then say:
            "I don't know the answer"
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            messages=messages, model=LLM_MODEL_NAME
        )
        return response.choices[0].message.content

    def llm_judge(self, question: str) -> dict:
        sys_message = {
            "role": "system",
            "content": "You are evaluating the retrieval quality of a RAG system. Always answer in json format",
        }
        message = {"role": "user", "content": question}

        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[sys_message, message],
            response_format={"type": "json_object"},
        )
        time.sleep(5)
        return json.loads(response.choices[0].message.content)

    def context_precision(self, question: str, retrieved_docs: list[str]) -> float:
        # Precision is number of relevant doc out of total retrieved docs
        relevant_chunk = 0
        if len(retrieved_docs) == 0:
            return 0.0
        for chunk in retrieved_docs:
            prompt = f"""
                Answer based on how relevant the question is based on the given retrieved doc. Mark it true if its relevant

                Question
                {question}

                doc:
                {chunk}

                Answer:
                {{
                    "relevant": true,
                    "description": "small reason"
                }}

                description will contain a short description on why this score
            """
            response = self.llm_judge(question=prompt)
            if response["relevant"]:
                relevant_chunk += 1

        return relevant_chunk / len(retrieved_docs)

    def context_recall(
        self, question: str, retrieved_docs: str, ground_truth: str
    ) -> float:
        # Actual relevant vectors/embedding out of expected embeddings
        # It tells whether context has enough information to find the correct answer
        prompt = f"""
            Does the context have enough information to find the correct answer to the given question. The answer should match with the given ground truth

            question
            {question}

            retrieved_docs:
            {retrieved_docs}

            ground_truth
            {ground_truth}

            response:
            {{
                "score": 0.5,
                "description": "short reason"
            }}

            scoring:
            1.0 = All important information needed for the answer is present.

            0.7 = Most important information is present, but some details are missing.

            0.5 = Some important information is present.

            0.0 = The required information is absent.
        """

        response = self.llm_judge(question=prompt)
        return float(response["score"])

    def evaluate_faithfulness(self, question, context, answer) -> float:
        # Whether LLM is faithful to the context and answer is according to the context only

        prompt = f"""
        Question:
        {question}

        Retrieved Context:
        {context}

        Generated Answer:
        {answer}

        Determine whether the claims in the generated answer are supported by the retrieved context.

        Return ONLY JSON:

        {{
            "score": 0.0,
            "reason": "short explanation"
        }}

        Scoring:

        1.0 = All claims are supported.

        0.7 = Mostly supported with minor issues.

        0.5 = Some claims are supported.

        0.0 = Unsupported or contradictory.
        """

        response = self.llm_judge(question=prompt)
        return float(response["score"])

    def evaluate_relevancy(self, question, answer) -> float:
        # Whether its actually the answer to the question
        prompt = f"""
        Question:
        {question}

        Generated Answer:
        {answer}

        Does the generated answer actually
        answer the question?

        Return ONLY JSON:

        {{
            "score": 0.0,
            "reason": "short explanation"
        }}

        Scoring:

        1.0 = Directly answers the question.

        0.7 = Mostly answers the question.

        0.5 = Partially answers the question.

        0.0 = Completely off-topic.
        """

        response = self.llm_judge(prompt)
        return float(response["score"])

    def evaluate_correctness(self, answer, ground_truth) -> float:
        # Whether the answer is actually the real answer according to ground truth
        prompt = f"""
            Generated Answer:
            {answer}

            Ground Truth Answer:
            {ground_truth}

            Determine whether the generated answer is factually correct compared with the ground truth.

            Return ONLY JSON:

            {{
                "score": 0.0,
                "reason": "short explanation"
            }}

            Scoring:

            1.0 = Completely correct.

            0.7 = Mostly correct with minor omissions.

            0.5 = Partially correct.

            0.0 = Incorrect or contradictory.
        """

        response = self.llm_judge(question=prompt)
        return float(response["score"])

    def evaluate_rag(self):
        with open("src/ai_engineering/week4/day1/golden_dataset.json", "r") as fp:
            golden_data_set = json.load(fp=fp)

        qdrant_client = Quadrant()
        qdrant_client.initialise(delete_colleciton=False)

        all_scores = {
            "precision": [],
            "recall": [],
            "faithfulness": [],
            "relevancy": [],
            "correctness": [],
        }

        for i, data in enumerate(golden_data_set):
            result = qdrant_client.retrieve(question=data["question"])
            retrieved_text: list[str] = []
            for point in result:
                # print(f"Score: {point.score}, id: {point.id}, payload: {point.payload}")
                retrieved_text.append(point.payload["text"])

            context = "\n".join(retrieved_text)
            all_scores["precision"].append(
                self.context_precision(
                    question=data["question"], retrieved_docs=retrieved_text
                )
            )
            all_scores["recall"].append(
                self.context_recall(
                    question=data["question"],
                    retrieved_docs=context,
                    ground_truth=data["ground_truth"],
                )
            )

            generated_answer = self.ask_llm(question=data["question"], context=context)
            print(f"Generated answer: {generated_answer}")

            all_scores["faithfulness"].append(
                self.evaluate_faithfulness(
                    question=data["question"], context=context, answer=generated_answer
                )
            )
            all_scores["correctness"].append(
                self.evaluate_correctness(
                    answer=generated_answer, ground_truth=data["ground_truth"]
                )
            )
            all_scores["relevancy"].append(
                self.evaluate_relevancy(
                    question=data["question"], answer=generated_answer
                )
            )

        print("=" * 60)
        print(all_scores)
        print("\n")
        print("-" * 70)
        for k, v in all_scores.items():
            print(f"{k}: {float(sum(v)/len(v))}")


def main():
    rag_obj = RAG()
    rag_obj.evaluate_rag()


if __name__ == "__main__":
    main()
