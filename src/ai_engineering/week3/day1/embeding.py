import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(array1, array2):
    return np.dot(array1, array2) / (np.linalg.norm(array1) * (np.linalg.norm(array2)))


def main():
    # s1 = "Civil engineering is fun"
    # s2 = "AI engineering is fun"

    s1 = "Apple"
    s2 = "Iphone"

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    v1 = model.encode(s1)
    v2 = model.encode(s2)

    print(cosine_similarity(array1=v1, array2=v2))


if __name__ == "__main__":
    main()
