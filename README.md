# **Arxiv Semantic Search - Setup Guide (macOS)**  

This guide provides step-by-step instructions to **set up and run** the **Arxiv Semantic Search** project on macOS. The system will:  

- Fetch research papers from **arXiv**  
- Process and store them in **Couchbase**  
- Generate **embeddings** using **sentence-transformers**  
- Perform **semantic search** to find the most relevant papers  
- Provide **paper links, summaries, and metadata**  

---

## **1. Install Prerequisites**  

### **1.1 Install Homebrew (If Not Installed)**  
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### **1.2 Install Required System Packages**  
```bash
brew install libcouchbase
brew install python3
```

---

## **2. Set Up Couchbase**  

### **2.1 Download and Install Couchbase Server**  
1. Download Couchbase Community Edition from the official website:  
   [https://www.couchbase.com/downloads](https://www.couchbase.com/downloads)  
2. Install the `.pkg` file and follow the on-screen instructions.  
3. Start Couchbase from **Applications** or run:  
   ```bash
   open -a Couchbase\ Server
   ```

### **2.2 Create a New Bucket (`arxiv`)**  
1. Open Couchbase Web Console at [http://localhost:8091](http://localhost:8091)  
2. Login using the default credentials:  
   - Username: `Administrator`  
   - Password: `password`  
3. Navigate to **Buckets** → Click **Create Bucket**  
   - Name: `arxiv`  
   - Memory Quota: `256MB`  
   - Click **Create Bucket**  

### **2.3 Create Indexes for Fast Search**  
1. Open the Couchbase Web Console  
2. Navigate to the **Query** tab  
3. Run the following SQL queries:  
   ```sql
   CREATE PRIMARY INDEX ON `arxiv`;
   CREATE INDEX `idx_embedding` ON `arxiv`(`embedding`);
   ```

---

## **3. Set Up the Project**  

### **3.1 Clone or Create the Project Directory**  
```bash
mkdir arxiv-semantic-search
cd arxiv-semantic-search
```

### **3.2 Set Up a Virtual Environment**  
```bash
python3 -m venv venv
source venv/bin/activate  # Activate the virtual environment
```

### **3.3 Create a `requirements.txt` File**  
Create a `requirements.txt` file in the project folder with the following content:
```
arxiv
requests
pdfplumber
torch
torchvision
transformers
sentence-transformers
couchbase
numpy
scipy
```

### **3.4 Install Dependencies from `requirements.txt`**  
```bash
pip install -r requirements.txt
```

---

## **4. Run the Full Application**  

### **4.1 Run the Script to Store Papers**  
```bash
streamlit run main.py
```
- The script will:  
  - Fetch research papers from **arXiv**  
  - Extract and store them in **Couchbase**  
  - Generate **embeddings** for semantic search  

### **4.2 Perform Semantic Search**  
Once the papers are stored, enter your query when prompted:  
```bash
Enter your search query: Transformer models for NLP
```
The system will return:  
- Paper Title  
- Paper URL  
- PDF Link  
- Summary  
- Similarity Score  

---

## **5. Example Output**  
```plaintext
Top Search Results:

[1] Advances in Deep Learning
URL: https://arxiv.org/abs/2402.00123
PDF Link: https://arxiv.org/pdf/2402.00123.pdf
Summary: This paper discusses advances in deep learning architectures...
Similarity Score: 0.8923
--------------------------------------------------------------------------------
[2] Efficient Neural Networks for Image Processing
URL: https://arxiv.org/abs/2402.00567
PDF Link: https://arxiv.org/pdf/2402.00567.pdf
Summary: We propose a CNN-based model for fast image processing...
Similarity Score: 0.8756
--------------------------------------------------------------------------------
```

---

## **6. Additional Notes**  

### **6.1 Restart Couchbase**  
If Couchbase stops working, restart it using:  
```bash
open -a Couchbase\ Server
```

### **6.2 Deactivate Virtual Environment**  
```bash
deactivate
```

### **6.3 Remove Virtual Environment (If Needed)**  
```bash
rm -rf venv
```

---

## **7. Next Steps**  

- Setup and fetch papers (**Completed**)  
- Store in Couchbase and generate embeddings (**Completed**)  
- Perform semantic search (**Completed**)  
- Build a Web API for search queries (**Completed**)  

---