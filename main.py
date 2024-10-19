from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from transformers import pipeline
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pdfplumber
import faiss
import numpy as np
from io import BytesIO
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Load the question-answering pipeline with a fine-tuned model
qa_pipeline = pipeline("question-answering", model="models/best_bert_model")
# qa_pipeline = pipeline("question-answering", model="fine_tuned_model_all_data")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define a Pydantic model for the question input
class Question(BaseModel):
    question: str

# Load embedding model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize FAISS index for the vector database
dimension = 384  # Based on the SentenceTransformer model's embedding size
index = faiss.IndexFlatL2(dimension)

# Store mapping between text chunks and embeddings
pdf_texts = []

# Load the SQuAD 2.0 dataset
squad_data = load_dataset("squad_v2")

# Convert the dataset into a dictionary of questions and their corresponding contexts
question_contexts = []
for entry in squad_data["train"]:
    question_contexts.append({
        "question": entry["question"],
        "context": entry["context"]
    })

# Function to retrieve the most relevant context from the SQuAD dataset
def get_relevant_context(question):
    # A simple matching function to find the most relevant context
    for qc in question_contexts:
        if qc["question"].lower() == question.lower():
            return qc["context"]
    
    # If no exact match is found, return the first result as a fallback
    return question_contexts[0]["context"]

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file: BytesIO):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Serve the chatbot UI (the HTML file)
@app.get("/", response_class=HTMLResponse)
async def get_chatbot_ui():
    # Serve the HTML file from the project directory
    with open("htmls/index_.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Endpoint to upload PDFs and add the content to the FAISS index
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Read PDF content
    pdf_content = await file.read()
    text = extract_text_from_pdf(BytesIO(pdf_content))

    # Check if text extraction worked
    if not text:
        return {"message": "Failed to extract text from the PDF."}
    
    print(f"Extracted text from PDF: {text[:200]}...")  # Print first 200 characters for debugging

    # Split text into smaller chunks (FAISS works better with smaller segments)
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]  # Split into 500 character chunks

    # Generate embeddings for each chunk and add them to FAISS index
    embeddings = embedding_model.encode(chunks)
    print(f"Generated {len(embeddings)} embeddings.")  # Debugging

    index.add(np.array(embeddings))  # Add to FAISS index
    pdf_texts.extend(chunks)  # Store the chunks for retrieval

    return {"message": "PDF uploaded and processed successfully!"}

# Endpoint to handle questions sent by the user
@app.post("/ask")
async def ask_question(q: Question):
    # Generate embedding for the question
    question_embedding = embedding_model.encode([q.question])
    
    print(f"Question: {q.question}")
    print("Generated question embedding:", question_embedding)
    
    # Check if FAISS index is empty
    if len(pdf_texts) == 0:
        print("FAISS index is empty. Falling back to SQuAD.")
        relevant_text = get_relevant_context(q.question)
    else:
        # Search FAISS index for the closest text chunk
        D, I = index.search(np.array(question_embedding), k=1)  # Search for the nearest chunk

        print(f"FAISS search result: Distances - {D}, Indices - {I}")

        # Handle case where no valid index is found (-1 is returned)
        if I[0][0] == -1 or D[0][0] > 1e38:  # No valid match found
            print("No relevant context found in PDF. Falling back to SQuAD.")
            relevant_text = get_relevant_context(q.question)
        else:
            # Retrieve the closest matching chunk of text from the PDF
            relevant_text = pdf_texts[I[0][0]]
            print(f"Found relevant context from PDF: {relevant_text[:200]}...")  # Print first 200 chars

    # Run the question-answering pipeline with the provided question and context
    result = qa_pipeline(question=q.question, context=relevant_text)
    
    print(f"Answer: {result['answer']}")  # Print answer for debugging
    
    return {"answer": result['answer'], "context": relevant_text}
