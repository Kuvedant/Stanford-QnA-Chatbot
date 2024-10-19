# rag_model.py

from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# Initialize the tokenizer, retriever, and model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-base")

# Use the FAISS index we created
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-base",
    index_name="custom",
    passages_path="embeddings.npy",
    index_path="documents.index"
)

model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-base", retriever=retriever)

def generate_answer(question, context):
    input_dict = tokenizer.prepare_seq2seq_batch([question], return_tensors="pt")
    generated = model.generate(**input_dict)
    answer = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return answer
