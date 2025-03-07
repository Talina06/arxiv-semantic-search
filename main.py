import io
import json
import arxiv
import requests
import pdfplumber
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from datetime import datetime
from scipy.spatial.distance import cosine
from couchbase.cluster import Cluster, ClusterOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.exceptions import CouchbaseException


class ArxivSemanticSearch:
    """Fetches PDFs from arXiv, processes them in memory, generates embeddings, stores JSON in Couchbase, and performs semantic search."""

    def __init__(self, couchbase_host="localhost", bucket_name="arxiv"):
        # Load a pre-trained sentence transformer model
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.client = arxiv.Client()

        # Connect to Couchbase
        self.cluster = Cluster(f"couchbase://{couchbase_host}", ClusterOptions(PasswordAuthenticator("Administrator", "password")))
        self.bucket = self.cluster.bucket(bucket_name)
        self.collection = self.bucket.default_collection()

        # Ensure required indexes exist
        self.ensure_indexes()

    def ensure_indexes(self):
        """Creates necessary indexes in Couchbase for querying embeddings."""
        try:
            query_service = self.cluster.query
            query_service("CREATE PRIMARY INDEX IF NOT EXISTS ON `arxiv`;")
            query_service("CREATE INDEX `idx_embedding` IF NOT EXISTS ON `arxiv`(`embedding`);")
            print("✅ Indexes are set up correctly.")
        except CouchbaseException as e:
            print(f"❌ Failed to create indexes: {e}")

    def fetch_papers(self, query, max_results=5):
        """Fetches research papers from arXiv."""
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
        return list(self.client.results(search))

    def fetch_pdf_in_memory(self, pdf_url):
        """Fetches a PDF from arXiv and loads it into memory."""
        response = requests.get(pdf_url, stream=True)
        return io.BytesIO(response.content) if response.status_code == 200 else None

    def extract_text_per_page(self, pdf_file):
        """Extracts structured text from every page of an in-memory PDF."""
        structured_data = {"pages": []}
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    structured_data["pages"].append({"page": page_num, "content": text})
        return structured_data

    def chunk_text(self, text, chunk_size=200):
        """Splits text into smaller chunks for embedding retrieval."""
        words = text.split()
        return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def generate_embeddings(self, text_chunks):
        """Generates embeddings for each text chunk."""
        return [self.embedding_model.encode(chunk).tolist() for chunk in text_chunks]

    def store_in_couchbase(self, paper_id, paper_metadata, embeddings):
        """Stores extracted JSON and embeddings in Couchbase."""
        try:
            self.collection.upsert(paper_id, paper_metadata)
            for idx, embedding in enumerate(embeddings):
                embedding_id = f"{paper_id}_chunk_{idx}"
                self.collection.upsert(embedding_id, {
                    "paper_id": paper_id,
                    "chunk_index": idx,
                    "embedding": embedding
                })
            print(f"✅ Stored in Couchbase: {paper_id}")
        except CouchbaseException as e:
            print(f"❌ Error storing {paper_id}: {e}")

    def process_papers(self, query, max_results=5):
        """Fetches papers, processes PDFs in memory, generates embeddings, and stores JSON in Couchbase."""
        papers = self.fetch_papers(query, max_results)

        for paper in papers:
            paper_id = paper.entry_id.split("/")[-1]
            pdf_file = self.fetch_pdf_in_memory(paper.pdf_url)

            if pdf_file:
                print(f"✅ Processing: {paper.title}")
                structured_text = self.extract_text_per_page(pdf_file)

                # Chunk text & generate embeddings
                all_chunks = []
                for page in structured_text["pages"]:
                    chunks = self.chunk_text(page["content"])
                    all_chunks.extend(chunks)

                embeddings = self.generate_embeddings(all_chunks)

                paper_metadata = {
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "summary": paper.summary,
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "updated": paper.updated.strftime("%Y-%m-%d"),
                    "url": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "structured_content": structured_text
                }

                self.store_in_couchbase(paper_id, paper_metadata, embeddings)
            else:
                print(f"❌ Failed to fetch PDF for: {paper.title}")

    def get_all_embeddings(self):
        """Retrieves all stored embeddings from Couchbase."""
        try:
            query = "SELECT META().id AS doc_id, embedding, paper_id, chunk_index FROM `arxiv` WHERE embedding IS NOT NULL"
            results = self.cluster.query(query)
            return [row for row in results]
        except CouchbaseException as e:
            print(f"❌ Error retrieving embeddings: {e}")
            return []

    def search(self, query_text, top_k=5):
        """Handles user query input and performs semantic search."""
        print(f"🔍 Searching for: {query_text}")
        query_embedding = self.embedding_model.encode(query_text)

        # Retrieve stored embeddings
        stored_embeddings = self.get_all_embeddings()

        # Compute similarity scores
        results = []
        for item in stored_embeddings:
            stored_vector = np.array(item["embedding"])
            similarity = 1 - cosine(query_embedding, stored_vector)  # Higher is better
            results.append((item["doc_id"], item["paper_id"], item["chunk_index"], similarity))

        # Sort by similarity score
        results.sort(key=lambda x: x[3], reverse=True)

        # Get top-K relevant papers and fetch metadata
        top_results = []
        for _, paper_id, _, score in results[:top_k]:
            try:
                doc_result = self.collection.get(paper_id)
                doc = doc_result.content_as[dict]  # Retrieve metadata

                top_results.append({
                    "title": doc.get("title", "[No Title]"),
                    "summary": doc.get("summary", "[No Summary]"),
                    "url": doc.get("url", "[No URL]"),
                    "pdf_url": doc.get("pdf_url", "[No PDF URL]"),
                    "similarity_score": score
                })
            except CouchbaseException:
                continue

        return top_results


# if __name__ == "__main__":
#     processor = ArxivSemanticSearch(couchbase_host="localhost", bucket_name="arxiv")
#
#     # Fetch and process papers
#     topic = "deep learning"  # Modify as needed
#     processor.process_papers(topic, max_results=3)
#
#     # Perform semantic search
#     user_query = input("Enter your search query: ")
#     search_results = processor.search(user_query, top_k=3)
#
#     print("\n🔎 **Top Search Results:**\n")
#     for idx, result in enumerate(search_results):
#         print(f"📄 **[{idx+1}] {result['title']}**")
#         print(f"🔗 [Read Paper]({result['url']})")
#         print(f"📄 **PDF Link:** {result['pdf_url']}")
#         print(f"📖 **Summary:** {result['summary']}")
#         print(f"🔢 Similarity Score: {result['similarity_score']:.4f}")
#         print("-" * 80)


# Streamlit UI
def main():
    st.title("Arxiv Semantic Search")

    # Input for search query
    query = st.text_input("Enter search query (e.g., 'deep learning'): ")

    if st.button("Search"):
        if query:
            processor = ArxivSemanticSearch(couchbase_host="localhost", bucket_name="arxiv")

            # Perform semantic search
            search_results = processor.search(query, top_k=3)

            # Display search results
            st.subheader("🔎 Top Search Results:")
            for idx, result in enumerate(search_results):
                st.markdown(f"**[{idx+1}] {result['title']}**")
                st.markdown(f"[Read Paper]({result['url']})")
                st.markdown(f"**PDF Link:** {result['pdf_url']}")
                st.markdown(f"**Summary:** {result['summary']}")
                st.markdown(f"**Similarity Score:** {result['similarity_score']:.4f}")
                st.markdown("-" * 80)
        else:
            st.error("Please enter a search query.")

    # Button to trigger paper processing (for demonstration)
    if st.button("Fetch and Process Papers"):
        processor = ArxivSemanticSearch(couchbase_host="localhost", bucket_name="arxiv")
        topic = "deep learning"  # Modify as needed
        processor.process_papers(topic, max_results=3)
        st.success("Papers processed successfully!")


if __name__ == "__main__":
    main()