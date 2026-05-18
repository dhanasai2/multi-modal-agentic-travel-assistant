import hashlib
from functools import lru_cache

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


CITY_FACTS: dict[str, list[str]] = {
    "paris": [
        "Paris is known for landmarks such as the Eiffel Tower, Louvre Museum, and Seine riverbanks.",
        "The city has dense metro connectivity and compact neighborhoods ideal for walking-based itineraries.",
        "Classic experiences include Montmartre, cafe culture, and evening river cruises.",
    ],
    "tokyo": [
        "Tokyo combines hyper-modern districts like Shibuya with historic temples like Senso-ji.",
        "Rail transit is precise and extensive, making neighborhood hopping highly efficient.",
        "Travelers often mix food markets, skyline viewpoints, and day trips to nearby nature areas.",
    ],
    "new york": [
        "New York City features iconic areas like Manhattan, Brooklyn, Central Park, and Times Square.",
        "The city supports diverse travel styles with museums, Broadway, food neighborhoods, and waterfronts.",
        "Subway access makes it practical to explore multiple boroughs on the same day.",
    ],
}


class HashEmbeddings(Embeddings):
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def _encode(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, len(digest), 2):
                idx = (digest[i] << 8 | digest[i + 1]) % self.dim
                vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


@lru_cache(maxsize=1)
def get_vector_store() -> FAISS:
    texts: list[str] = []
    metadatas: list[dict] = []
    for city, facts in CITY_FACTS.items():
        for fact in facts:
            texts.append(fact)
            metadatas.append({"city": city})
    return FAISS.from_texts(texts=texts, embedding=HashEmbeddings(), metadatas=metadatas)


def city_exists(city: str) -> bool:
    return city.strip().lower() in CITY_FACTS


def retrieve_city_facts(city: str, k: int = 4) -> str:
    store = get_vector_store()
    key = city.strip().lower()
    docs = store.similarity_search(f"{city} travel facts", k=k)
    filtered = [doc.page_content for doc in docs if doc.metadata.get("city") == key]
    if not filtered:
        return ""
    return " ".join(filtered)


def vector_city_list() -> list[str]:
    return sorted(CITY_FACTS.keys())
