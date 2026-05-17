import chromadb


class ChromaStore:
    def __init__(self, client=None, collection_name: str = "articles"):
        if client is None:
            client = chromadb.PersistentClient(path="storage/chroma_db")
        self._client = client
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def insert_chunks(
        self,
        article_id: int,
        story_group_id: int,
        source_name: str,
        published_at: str,
        field_texts: dict,
        field_embeddings: dict,
    ) -> None:
        ids, documents, embeddings_list, metadatas = [], [], [], []
        for field, text in field_texts.items():
            if field not in field_embeddings:
                continue
            ids.append(f"{article_id}_{field}")
            documents.append(text)
            embeddings_list.append(field_embeddings[field])
            metadatas.append({
                "field": field,
                "article_id": str(article_id),
                "story_group_id": str(story_group_id),
                "source_name": source_name,
                "published_at": published_at,
            })
        if ids:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings_list,
                metadatas=metadatas,
            )

    def get_summaries_since(self, cutoff_date: str) -> list[dict]:
        results = self._collection.get(
            where={"field": "summary"},
            include=["metadatas", "embeddings"],
        )
        out = []
        for meta, emb in zip(results["metadatas"], results["embeddings"]):
            if meta["published_at"] >= cutoff_date:
                out.append({
                    "article_id": int(meta["article_id"]),
                    "story_group_id": int(meta["story_group_id"]),
                    "embedding": list(emb),
                })
        return out

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["metadatas", "documents", "distances"],
        }
        if where:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)
        out = []
        for i in range(len(results["ids"][0])):
            out.append({
                "article_id": int(results["metadatas"][0][i]["article_id"]),
                "field": results["metadatas"][0][i]["field"],
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            })
        return out
