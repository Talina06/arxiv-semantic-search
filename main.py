import streamlit as st
import numpy as np
import logging
from collections import defaultdict
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster, ClusterOptions

# ---------------------- Couchbase Connection ---------------------- #
class CouchbaseDB:
    """Handles Couchbase connection and operations."""

    def __init__(self, host="localhost", bucket_name="arxiv"):
        self.cluster = Cluster(f"couchbase://{host}",
                               ClusterOptions(PasswordAuthenticator("Administrator", "password")))
        self.bucket = self.cluster.bucket(bucket_name)
        self.collection = self.bucket.default_collection()

    def get_all_embeddings(self):
        """Retrieve all stored chunk embeddings."""
        query = "SELECT META().id AS doc_id, embedding, paper_id FROM `arxiv` WHERE embedding IS NOT NULL"
        return [row for row in self.cluster.query(query)]

    def get_document(self, doc_id):
        """Fetch the full paper metadata by ID."""
        try:
            return self.collection.get(doc_id).content_as[dict]
        except Exception:
            return None


# ---------------------- Semantic Search ---------------------- #
class SemanticSearcher:
    """Performs semantic search and ranks results by highest similarity score."""

    def __init__(self, db):
        self.db = db
        self.embedding_model = SentenceTransformer("Talina06/arxiv-search")

    def search(self, query_text, top_k=5):
        """Search for papers and return the highest similarity per paper along with metadata."""
        query_embedding = self.embedding_model.encode(query_text)
        stored_embeddings = self.db.get_all_embeddings()

        paper_matches = defaultdict(lambda: {"highest_similarity": 0})

        for item in stored_embeddings:
            paper_id = item["paper_id"]
            stored_embedding = np.array(item["embedding"])
            similarity = 1 - cosine(query_embedding, stored_embedding)

            if similarity > paper_matches[paper_id]["highest_similarity"]:
                paper_matches[paper_id]["highest_similarity"] = similarity

        return sorted(paper_matches.items(), key=lambda x: x[1]["highest_similarity"], reverse=True)[:top_k]


# ---------------------- Run Streamlit ---------------------- #
# Function to display paper details nicely
def display_search_results(results, db):
    st.subheader("🔎 Top Search Results:")

    for idx, (paper_id, data) in enumerate(results):
        paper = db.get_document(paper_id)

        # Display the paper title with a clickable link
        st.markdown(f"#### **[{paper['title']}]({paper['url']})**")

        # Display the summary (or placeholder if no summary exists)
        summary = paper.get('summary', 'No summary available for this paper.')
        pdf_link = paper.get('pdf_url', 'No PDF link available.')
        st.markdown(f"**Summary:** {summary}... [Read More]({pdf_link})")

        # Display similarity score
        st.markdown(f"**Similarity Score:** {data['highest_similarity']:.4f}")
        st.markdown("---")  # Horizontal line for separation


# Streamlit UI
st.title("🔎 Arxiv Semantic Search")

query = st.text_input("Enter search query:")
if st.button("Search") and query:
    db = CouchbaseDB()
    searcher = SemanticSearcher(db)
    results = searcher.search(query, top_k=5)

    display_search_results(results, db)
