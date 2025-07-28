# RAG Q&A Chatbot for Loan Approval Dataset

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about the [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction?select=Training+Dataset.csv) using document retrieval and a local generative AI model.

## Features
- **Retrieval-Augmented Generation:** Combines semantic search with a small language model for intelligent answers.
- **Local & Free:** Uses open-source models from Hugging Face, no API keys or paid credits required.
- **CLI Interface:** Ask questions about the dataset directly from your terminal.

## Setup Instructions

### 1. Clone the Repository
Place all files in a folder (e.g., `Assignment_08`).

### 2. Download the Dataset
- Download `Training Dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction?select=Training+Dataset.csv).
- Place the CSV file in the same folder as `rag_chatbot.py`.

### 3. Install Dependencies
Run the following command in your terminal:
```bash
pip install -r requirements.txt
```

### 4. Run the Chatbot
```bash
python rag_chatbot.py
```

## Usage
- After running the script, type your questions about the loan dataset.
- Type `exit` or `quit` to stop the chatbot.

## Example Questions
- "What is the most common reason for loan rejection?"
- "How many applicants are self-employed?"
- "What is the average loan amount?"

## Notes
- The chatbot uses each row of the dataset as a document for retrieval.
- The generative model is `google/flan-t5-small` (runs locally).

## Troubleshooting
- Make sure you have enough RAM (embedding and LLM models are lightweight, but may require ~2GB RAM).
- If you encounter errors about missing files, check that `Training Dataset.csv` is in the correct folder.

## License
This project is for educational purposes only. Dataset © Kaggle contributors. 