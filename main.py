import io
import json
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
import logging
import arxiv

# Logging setup
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),  # To output logs to the console
        logging.FileHandler("arxiv_semantic_search.log")  # To save logs to a file
    ]
)
logger = logging.getLogger()


class ArxivSemanticSearch:
    """Fetches PDFs from arXiv, processes them in memory, generates embeddings, stores JSON in Couchbase, and performs semantic search."""

    def __init__(self, couchbase_host="localhost", bucket_name="arxiv"):
        logger.info("Initializing ArxivSemanticSearch...")

        # Load a pre-trained sentence transformer model
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("SentenceTransformer model loaded.")

        # Connect to Couchbase
        try:
            self.cluster = Cluster(f"couchbase://{couchbase_host}",
                                   ClusterOptions(PasswordAuthenticator("Administrator", "password")))
            self.bucket = self.cluster.bucket(bucket_name)
            self.collection = self.bucket.default_collection()
            logger.info(f"Connected to Couchbase on {couchbase_host}.")
        except CouchbaseException as e:
            logger.error(f"Failed to connect to Couchbase: {e}")
            raise

        # Ensure required indexes exist
        self.ensure_indexes()

    def ensure_indexes(self):
        """Creates necessary indexes in Couchbase for querying embeddings and feedback."""
        try:
            query_service = self.cluster.query
            query_service("CREATE PRIMARY INDEX IF NOT EXISTS ON `arxiv`;")
            query_service("CREATE INDEX `idx_embedding` IF NOT EXISTS ON `arxiv`(`embedding`);")
            logger.info("Indexes are set up correctly.")
        except CouchbaseException as e:
            logger.error(f"Failed to create indexes: {e}")

    def fetch_papers(self, query, max_results=5):
        """Fetches research papers from arXiv."""
        try:
            search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
            papers = list(search.results())
            logger.info(f"Fetched {len(papers)} papers for query: {query}")
            return papers
        except Exception as e:
            logger.error(f"Error fetching papers: {e}")
            return []

    def fetch_pdf_in_memory(self, pdf_url):
        """Fetches a PDF from arXiv and loads it into memory."""
        response = requests.get(pdf_url, stream=True)
        return io.BytesIO(response.content) if response.status_code == 200 else None

    def extract_text_per_page(self, pdf_file):
        """Extracts structured text from every page of an in-memory PDF."""
        structured_data = {"pages": []}
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text:
                        structured_data["pages"].append({"page": page_num, "content": text})
            logger.info("PDF text extraction complete.")
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
        return structured_data

    def chunk_text(self, text, chunk_size=200):
        """Splits text into smaller chunks for embedding retrieval."""
        words = text.split()
        chunks = [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]
        logger.debug(f"Text chunked into {len(chunks)} parts.")
        return chunks

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
            logger.info(f"Stored embeddings for {paper_id}.")
        except CouchbaseException as e:
            logger.error(f"Error storing {paper_id} in Couchbase: {e}")

    def store_feedback_in_couchbase(self, paper_id, feedback, query):
        """Stores user feedback for a paper in Couchbase."""
        feedback_id = f"feedback_{paper_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        feedback_data = {
            "paper_id": paper_id,
            "feedback": feedback,
            "query": query,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            self.collection.upsert(feedback_id, feedback_data)
            logger.info(f"Feedback stored for paper {paper_id} with feedback: {feedback}")
        except CouchbaseException as e:
            logger.error(f"Error storing feedback for {paper_id}: {e}")

    def store_query_feedback_in_couchbase(self, query, feedback=None):
        """Stores feedback for the entire search query (including default 'None')."""
        feedback_id = f"query_feedback_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        feedback_data = {
            "query": query,
            "feedback": feedback if feedback is not None else "None",  # Default to 'None' if no feedback provided
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            self.collection.upsert(feedback_id, feedback_data)
            logger.info(f"Search query feedback stored with feedback: {feedback if feedback else 'None'}")
        except CouchbaseException as e:
            logger.error(f"Error storing query feedback: {e}")

    def process_papers(self, query, max_results=5):
        """Fetches papers, processes PDFs in memory, generates embeddings, and stores JSON in Couchbase."""
        papers = self.fetch_papers(query, max_results)

        for paper in papers:
            paper_id = paper.entry_id.split("/")[-1]
            pdf_file = self.fetch_pdf_in_memory(paper.pdf_url)

            if pdf_file:
                logger.info(f"Processing paper: {paper.title}")
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
                logger.error(f"Failed to fetch PDF for paper: {paper.title}")

    def get_all_embeddings(self):
        """Retrieves all stored embeddings from Couchbase."""
        try:
            query = "SELECT META().id AS doc_id, embedding, paper_id, chunk_index FROM `arxiv` WHERE embedding IS NOT NULL"
            results = self.cluster.query(query)
            return [row for row in results]
        except CouchbaseException as e:
            logger.error(f"Error retrieving embeddings: {e}")
            return []

    def search(self, query_text, top_k=5):
        """Handles user query input and performs semantic search."""
        logger.info(f"Performing search for query: {query_text}")
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

        logger.info(f"Top {top_k} search results returned.")
        return top_results

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
                st.markdown(f"**[{idx + 1}] {result['title']}**")
                st.markdown(f"[Read Paper]({result['url']})")
                st.markdown(f"**PDF Link:** {result['pdf_url']}")
                st.markdown(f"**Summary:** {result['summary']}")
                st.markdown(f"**Similarity Score:** {result['similarity_score']:.4f}")
                st.markdown("-" * 80)

                # Feedback for each result
                feedback = st.radio(
                    f"Was the result for **{result['title']}** useful?",
                    options=["please select", "👍", "👎"],
                    key=f"feedback_{idx}",  # Unique key for each feedback
                    horizontal=True
                )

                # Store feedback when the user submits
                if st.button(f"Submit Feedback for {result['title']}", key=f"submit_{idx}"):
                    if feedback != "please select":
                        processor.store_feedback_in_couchbase(result["paper_id"], feedback, query)
                        st.success(f"Feedback for '{result['title']}' has been recorded as: {feedback}")
                    else:
                        st.warning(f"Please select feedback (👍 or 👎) for '{result['title']}'.")

        else:
            st.error("Please enter a search query.")

    # Button to trigger paper processing (for demonstration)
    if st.button("Fetch and Process Papers"):
        processor = ArxivSemanticSearch(couchbase_host="localhost", bucket_name="arxiv")
        topic = "deep learning"
        processor.process_papers(topic, max_results=3)
        st.success("Papers processed successfully!")

if __name__ == "__main__":
    main()
