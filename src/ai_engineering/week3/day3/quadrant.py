from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams
from sentence_transformers import SentenceTransformer

from ai_engineering.constants import (
    GROQ_API_KEY,
    LLM_MODEL_NAME,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_EMBEDDING_SIZE,
    QDRANT_ENDPOINT,
)


class Quadrant:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)
        self.sentence_transformer = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def initialize_qdrant_db(self, delete_existing: bool = False):
        if delete_existing:
            if self.client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
                self.client.delete_collection(QDRANT_COLLECTION_NAME)

            self.client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=QDRANT_EMBEDDING_SIZE,
                    distance=Distance.COSINE,
                ),
            )

        print(
            f"Created collection: {QDRANT_COLLECTION_NAME} with size {QDRANT_EMBEDDING_SIZE} and distance {Distance.COSINE}"
        )

    def upsert(self):
        self.initialize_qdrant_db(delete_existing=False)
        with open("knowledge.txt", "r") as fp:
            documents = [line.strip() for line in fp if line.strip()]

        print(f"Document array created with size {len(documents)}")

        embeded_docs = self.sentence_transformer.encode(documents)

        points: list[PointStruct] = []
        for i, e_doc in enumerate(embeded_docs):
            point = PointStruct(
                id=i + 1, vector=list(e_doc), payload={"text": documents[i]}
            )
            points.append(point)

        print(f"Created Vector Point struct with size {len(points)}")

        self.client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)
        print(f"Uploaded {len(points)} points to Qdrant")


class RAG:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def run_llm(self, query: str, context: str) -> str:
        sys_msg = {
            "role": "system",
            "content": f"Only answer based on the given context and do not hallucinate. context: {context}",
        }

        user_msg = {"role": "user", "content": query}

        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME, messages=[sys_msg, user_msg]
        )
        return response.choices[0].message.content

    def retrieve_context(self, query: str, top_k: int) -> list[ScoredPoint]:
        qdrant = Quadrant()
        # qdrant.upsert()

        embeded_query = qdrant.sentence_transformer.encode(query)

        result = qdrant.client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=list(embeded_query),
            limit=top_k,
            with_payload=True,
        ).points

        for data in result:
            print(f"Score: {data.score}, id: {data.id}, payload: {data.payload}")
        return result


def main():
    rag_obj = RAG()
    question = "Will i get any work from home opportunity ?"
    contexts: list[ScoredPoint] = rag_obj.retrieve_context(query=question, top_k=2)

    context = "\n".join(cont.payload["text"] for cont in contexts)

    print(rag_obj.run_llm(query=question, context=context))


if __name__ == "__main__":
    main()
