<<<<<<< HEAD
# Stanford-QnA-Chatbot
=======

# SQuAD QA Chatbot with RAG and Vector Database

This repository contains the implementation of a Question Answering (QA) system using the SQuAD 2.0 dataset. The model is based on Hugging Face's transformers library, utilizing a fine-tuned BERT model for extractive question answering. The project also incorporates Retrieval-Augmented Generation (RAG) and a vector database for real-time question answering with multi-turn conversations.

![SQuADBot - Your Question Answering Companion](Screenshot_SQuADBot.png)

## Project Overview

The goal of this project is to create a robust QA chatbot capable of answering questions from a context, integrating a vector database for retrieval and RAG for generation, and supporting multi-turn conversations.

## Key Features
- **Data Preprocessing and Fine-Tuning**: Tokenization of the SQuAD 2.0 dataset and fine-tuning of a pre-trained BERT model.
- **RAG and Vector Database Integration**: For efficient real-time retrieval of relevant passages, the system incorporates a vector database (e.g., FAISS) and RAG framework.
- **Multi-turn Conversations**: Contextualized QA, where the chatbot maintains conversation history and provides context-aware answers.
- **Visualization and Model Insights**: Visualizations of model confidence, answer start positions, answer lengths, and attention maps for interpretability.

## Installation

```bash
# Install necessary packages
!pip install "numpy<2.0" torch torchvision torchaudio transformers datasets accelerate nltk rouge_score tqdm
```

```bash
# Additional packages
pip install matplotlib seaborn scikit-learn
```

## Data Setup

The project uses the SQuAD 2.0 dataset, which can be downloaded and preprocessed as follows:

```python
from datasets import load_dataset
squad_dataset = load_dataset('squad_v2')
```

## Model Training

The BERT model is fine-tuned on the SQuAD 2.0 dataset using Hugging Face's Trainer API. Mixed precision is used when CUDA is available for faster training:

```python
import torch
from transformers import BertForQuestionAnswering, Trainer, TrainingArguments

# Load the model
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    per_device_train_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available()
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_validation_dataset
)

# Start training
trainer.train()
```

## Model Evaluation

The evaluation is performed using the validation set of SQuAD 2.0, and metrics such as Exact Match (EM), F1 score, BLEU, and ROUGE are computed.

```python
from evaluate import load
metric = load("squad_v2")
final_score = metric.compute(predictions=predictions, references=references)
```

## Visualizations

Several visualizations are created to understand the model's performance:

- **Training Loss vs Validation Loss**: Helped track the training process and identify overfitting.
- **Confusion Matrix**: Showed the model's performance in predicting answerable and unanswerable questions.
- **Answer Start Position Distribution**: Provided insights into the model's predictions in terms of where it believed the answer starts in the context.
- **Exact Match vs F1 Score**: Correlation between EM and F1 scores helped evaluate the trade-offs between strict matching and partial matching.
- **Answer Length Distribution**: Compared predicted and true answer lengths, helping identify discrepancies.
- **Confidence Distribution**: Visualized the confidence of the model in its predictions.

These visualizations provided a comprehensive understanding of the model's behavior and areas for improvement.

## Conclusion

This project successfully built a QA chatbot using a fine-tuned BERT model, integrated with RAG and a vector database for real-time retrieval and multi-turn conversations. The visualizations further aided in understanding model behavior and performance, leading to better fine-tuning and adjustments.

## License

This project is licensed under the MIT License.
>>>>>>> Initial commit for LLM based Chatbot on Squad 2 data
