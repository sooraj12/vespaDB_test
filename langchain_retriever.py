from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List
from vespa.io import VespaQueryResponse
from vespa.application import Vespa
from app import app


class VespaStreamingHybridRetriever(BaseRetriever):

    app: Vespa
    user: str
    pages: int = 5
    chunks_per_page: int = 3
    chunk_similarity_threshold: float = 0.8

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # targetHits = self.pages * self.chunks_per_page
        yql_query = """select 
                        id, 
                        url, 
                        title, 
                        page, 
                        authors, 
                        chunks 
                        from pdf 
                        where userQuery() or ({targetHits:1}nearestNeighbor(embedding,q))"""
        response: VespaQueryResponse = self.app.query(
            yql=yql_query,
            groupname=self.user,
            ranking="hybrid",
            query=query,
            hits=self.pages,
            body={
                "presentation.format.tensors": "short-value",
                "input.query(q)": f'embed(e5, "query: {query} ")',
            },
        )

        if not response.is_successful():
            raise ValueError(
                f"Query failed with status code {response.status_code}, url={response.url} response={response.json}"
            )
        return self._parse_response(response)

    def _parse_response(self, response: VespaQueryResponse) -> List[Document]:
        documents: List[Document] = []
        for hit in response.hits:
            fields = hit["fields"]
            chunks_with_scores = self._get_chunk_similarities(fields)
            ## Best k chunks from each page
            best_chunks_on_page = " ### ".join(
                [
                    chunk
                    for chunk, score in chunks_with_scores[0 : self.chunks_per_page]
                    if score > self.chunk_similarity_threshold
                ]
            )
            documents.append(
                Document(
                    id=fields["id"],
                    page_content=best_chunks_on_page,
                    title=fields["title"],
                    metadata={
                        "title": fields["title"],
                        "url": fields["url"],
                        "page": fields["page"],
                        "authors": fields["authors"],
                        "features": fields["matchfeatures"],
                    },
                )
            )
        return documents

    def _get_chunk_similarities(self, hit_fields: dict) -> List[tuple]:
        match_features = hit_fields["matchfeatures"]
        similarities = match_features["similarities"]
        chunk_scores = []
        for i in range(0, len(similarities)):
            chunk_scores.append(similarities.get(str(i), 0))
        chunks = hit_fields["chunks"]
        chunks_with_scores = list(zip(chunks, chunk_scores))
        return sorted(chunks_with_scores, key=lambda x: x[1], reverse=True)


hybrid_retriever = VespaStreamingHybridRetriever(
    app=app, user="test_user", pages=1, chunks_per_page=1
)

print(hybrid_retriever.invoke("what is the maxsim operator in colbert?"))
