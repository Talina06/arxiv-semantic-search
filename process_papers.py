import io
import logging
import requests
import pdfplumber
import arxiv
import concurrent.futures
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster, ClusterOptions

# ---------------------- Logging Setup ---------------------- #
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] - %(message)s",
    level=logging.INFO,  # Log at DEBUG level for detailed logs
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("arxiv_processing.log")
    ]
)
logger = logging.getLogger()


# ---------------------- Couchbase Connection ---------------------- #
class CouchbaseDB:
    """Handles Couchbase connection and operations."""

    def __init__(self, host="localhost", bucket_name="arxiv"):
        logger.info("Connecting to Couchbase...")
        try:
            self.cluster = Cluster(f"couchbase://{host}",
                                   ClusterOptions(PasswordAuthenticator("Administrator", "password")))
            self.bucket = self.cluster.bucket(bucket_name)
            self.collection = self.bucket.default_collection()
            logger.info(f"Connected to Couchbase on {host}.")
        except Exception as e:
            logger.error(f"Failed to connect to Couchbase: {e}")
            raise

    def store_document(self, doc_id, data):
        """Stores JSON data in Couchbase."""
        try:
            self.collection.upsert(doc_id, data)
            logger.debug(f"Stored document {doc_id}")
        except Exception as e:
            logger.warning(f"Error storing {doc_id}: {e}")

    def store_embedding(self, doc_id, embedding_data):
        """Stores chunk embeddings in Couchbase."""
        try:
            self.collection.upsert(doc_id, embedding_data)
            logger.debug(f"Stored embedding {doc_id}")
        except Exception as e:
            logger.warning(f"Error storing embedding {doc_id}: {e}")


# ---------------------- Summarization ---------------------- #
class ArxivSummarizer:
    """Summarizes research papers using Talina06/arxiv-summarization."""

    def __init__(self):
        logger.info("Loading summarization model...")
        self.model = T5ForConditionalGeneration.from_pretrained("Talina06/arxiv-summarization")
        self.tokenizer = T5Tokenizer.from_pretrained("Talina06/arxiv-summarization")
        logger.info("Summarization model loaded.")

    def summarize(self, text, max_length=150):
        """Generates a summary for the given text."""
        logger.debug("Summarizing text...")
        inputs = self.tokenizer("Summarize: " + text, return_tensors="pt", truncation=True, max_length=512)
        summary_ids = self.model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logger.debug(f"Generated summary: {summary}")
        return summary

    def summarize_full_text(self, full_text):
        """Uses a sliding window + adaptive summarization approach."""
        logger.debug("Splitting text into chunks for summarization...")
        text_chunks = self.chunk_text(full_text, chunk_size=512, stride=256)
        summaries = [self.summarize(chunk) for chunk in text_chunks]

        # Merge into a final summary
        final_summary = " ".join(summaries)  # Combine the summaries of all chunks

        # Limit the final summary length to 500 characters, or stop at the next full stop
        if len(final_summary) > 2000:
            final_summary = final_summary[:1000]
            last_full_stop = final_summary.rfind('.')

            # If a full stop exists, cut at the full stop
            if last_full_stop != -1:
                final_summary = final_summary[:last_full_stop + 1]

        logger.debug(f"Final Summary: {final_summary}")
        return final_summary

    def chunk_text(self, text, chunk_size=512, stride=256):
        """Splits text into chunks using a rolling window approach."""
        words = text.split()
        chunks = [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), stride)]
        logger.debug(f"Text split into {len(chunks)} chunks.")
        return chunks


# ---------------------- Paper Processing ---------------------- #
class ArxivPaperProcessor:
    """Fetches PDFs from arXiv, processes them, extracts text, summarizes, generates embeddings, and stores them."""

    def __init__(self, db, summarizer):
        self.db = db
        self.summarizer = summarizer
        self.embedding_model = SentenceTransformer("Talina06/arxiv-search")

    def fetch_papers_by_category(self, categories, max_results=5):
        """Fetches research papers from arXiv based on specific categories."""
        logger.info(f"📡 Fetching papers for categories: {', '.join(categories)}")
        try:
            search = arxiv.Search(
                query=" OR ".join(f"cat:{cat}" for cat in categories),  # Query by category
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            papers = list(search.results())
            logger.debug(f"Fetched {len(papers)} papers.")
            return papers
        except Exception as e:
            logger.warning(f"Error fetching papers: {e}")
            return []

    def fetch_pdf(self, pdf_url):
        """Fetches a PDF from ArXiv and loads it into memory."""
        logger.debug(f"Fetching PDF from URL: {pdf_url}")
        try:
            response = requests.get(pdf_url, stream=True)
            if response.status_code == 200:
                return io.BytesIO(response.content)
            else:
                logger.warning(f"Failed to fetch PDF: {pdf_url}")
                return None
        except Exception as e:
            logger.warning(f"Error fetching PDF: {e}")
            return None

    def extract_text(self, pdf_file):
        """Extracts text from a PDF."""
        logger.debug("Extracting text from PDF...")
        structured_data = {"pages": []}
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text:
                        structured_data["pages"].append({"page": page_num, "content": text})
            logger.debug("Text extraction completed.")
            return structured_data
        except Exception as e:
            logger.warning(f"Error extracting text: {e}")
            return structured_data

    def chunk_text(self, text, chunk_size=200):
        """Splits text into smaller chunks for embedding retrieval."""
        logger.debug("Chunking text...")
        words = text.split()
        chunks = [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]
        logger.debug(f"Text split into {len(chunks)} chunks.")
        return chunks

    def generate_embeddings(self, text_chunks):
        """Generates embeddings for text chunks."""
        logger.debug("Generating embeddings for text chunks...")
        embeddings = [self.embedding_model.encode(chunk).tolist() for chunk in text_chunks]
        logger.debug(f"Generated embeddings for {len(text_chunks)} chunks.")
        return embeddings

    def process_single_paper(self, paper):
        """Processes a single paper: Fetches, extracts, summarizes, generates embeddings, and stores it."""
        logger.debug(f"Processing paper: {paper.title}")
        paper_id = paper.entry_id.split("/")[-1]
        pdf_file = self.fetch_pdf(paper.pdf_url)

        if pdf_file:
            structured_text = self.extract_text(pdf_file)
            full_text = " ".join([page["content"] for page in structured_text["pages"] if page["content"]])

            # Summarization
            summary = self.summarizer.summarize(full_text)

            # Chunk text & generate embeddings
            text_chunks = self.chunk_text(full_text)
            embeddings = self.generate_embeddings(text_chunks)

            # Store paper metadata
            paper_metadata = {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "published": paper.published.strftime("%Y-%m-%d"),
                "url": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "summary": summary
            }
            self.db.store_document(paper_id, paper_metadata)

            # Store embeddings
            for idx, embedding in enumerate(embeddings):
                embedding_id = f"{paper_id}_chunk_{idx}"
                self.db.store_embedding(embedding_id, {
                    "paper_id": paper_id,
                    "chunk_index": idx,
                    "embedding": embedding
                })

            logger.info(f"✅ Processed: {paper.title} | Chunks Stored: {len(embeddings)}")
        else:
            logger.warning(f"⚠️ Failed to fetch PDF for: {paper.title}")

    def process_papers(self, categories, max_results=5, num_threads=5):
        """Fetches, processes, summarizes, and stores research papers and embeddings using multithreading."""
        logger.info("Starting paper processing...")
        papers = self.fetch_papers_by_category(categories, max_results)

        # Process papers in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(self.process_single_paper, papers)

        logger.info("✅ All paper processing complete.")


# ---------------------- Run Processing ---------------------- #
if __name__ == "__main__":
    db = CouchbaseDB()
    summarizer = ArxivSummarizer()
    processor = ArxivPaperProcessor(db, summarizer)

    categories = ["cs.LG", "cs.GR", "math.NA", "physics.comp-ph"]  # Customize as needed
    processor.process_papers(categories, max_results=20, num_threads=10)

    logger.info("✅ Multithreaded paper processing complete.")
