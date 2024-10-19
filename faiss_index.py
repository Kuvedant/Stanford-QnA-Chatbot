# faiss_index.py

import faiss
from transformers import BertTokenizerFast, BertModel
import numpy as np
import torch

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Let's assume we have a list of documents
documents = ["Document 1 text...", "Document 2 text...", "..."]

def embed(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:,0,:].numpy()
    return embeddings

# Build the index
embeddings = embed(documents)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index
faiss.write_index(index, 'documents.index')
np.save('embeddings.npy', embeddings)
