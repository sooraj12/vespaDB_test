import hashlib
import unicodedata

from typing import Iterable
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vespa.io import VespaResponse
from sample_pdfs import sample_pdfs
from app import app

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


docs_to_feed = []
for pdf in sample_pdfs():
    url = pdf["url"]
    loader = PyPDFLoader(url)
    pages = loader.load_and_split()
    for index, page in enumerate(pages):
        source = page.metadata["source"]
        chunks = text_splitter.transform_documents([page])
        text_chunks = [chunk.page_content for chunk in chunks]
        text_chunks = [remove_control_characters(chunk) for chunk in text_chunks]
        page_number = index + 1
        vespa_id = f"{url}#{page_number}"
        hash_value = hashlib.sha1(vespa_id.encode()).hexdigest()
        fields = {
            "title": pdf["title"],
            "url": url,
            "page": page_number,
            "id": hash_value,
            "authors": [a.strip() for a in pdf["authors"].split(",")],
            "chunks": text_chunks,
            "metadata": page.metadata,
        }
        docs_to_feed.append(fields)


def vespa_feed(user: str) -> Iterable[dict]:
    for doc in docs_to_feed:
        yield {"fields": doc, "id": doc["id"], "groupname": user}


def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(
            f"Document {id} failed to feed with status code {response.status_code}, url={response.url} response={response.json}"
        )


app.feed_iterable(
    schema="pdf", iter=vespa_feed("test_user"), namespace="personal", callback=callback
)
