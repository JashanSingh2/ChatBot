import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import kagglehub

# 1. Download the dataset if not present
DATASET_ID = "sonalisingh1411/loan-approval-prediction"
DATA_FILENAME = "Training Dataset.csv"

if not os.path.exists(DATA_FILENAME):
    print("Downloading dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download(DATASET_ID)
    # Find the correct file in the downloaded dataset
    for root, dirs, files in os.walk(dataset_path):
        if DATA_FILENAME in files:
            data_file_path = os.path.join(root, DATA_FILENAME)
            break
    else:
        raise FileNotFoundError(f"{DATA_FILENAME} not found in downloaded dataset.")
else:
    data_file_path = DATA_FILENAME

print(f"Using dataset file: {data_file_path}")
df = pd.read_csv(data_file_path)

# 2. Prepare documents (each row as a document)
documents = []
for idx, row in df.iterrows():
    doc = f"Row {idx+1}: " + ", ".join([f"{col}: {row[col]}" for col in df.columns])
    documents.append(doc)

# 3. Load embedding model
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 4. Embed all documents
print("Embedding documents...")
doc_embeddings = embedder.encode(documents, show_progress_bar=True, convert_to_numpy=True)

# 5. Build FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# 6. Load generative model (small LLM)
print("Loading generative model...")
llm_name = 'google/flan-t5-small'
tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForSeq2SeqLM.from_pretrained(llm_name)

def retrieve(query, k=3):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, k)
    return [documents[i] for i in I[0]]

def generate_answer(question, retrieved_docs):
    context = "\n".join(retrieved_docs)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = llm.generate(**inputs, max_new_tokens=128)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

# 7. CLI loop
print("\nRAG Q&A Chatbot is ready! Ask questions about the loan dataset. Type 'exit' to quit.\n")
while True:
    user_q = input("You: ")
    if user_q.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break
    retrieved = retrieve(user_q, k=3)
    answer = generate_answer(user_q, retrieved)
    print(f"Bot: {answer}\n") 