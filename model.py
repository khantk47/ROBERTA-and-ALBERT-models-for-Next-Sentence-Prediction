import torch
from transformers import BertTokenizer, BertForNextSentencePrediction, AlbertForMaskedLM, RobertaForMaskedLM, AlbertTokenizer, RobertaTokenizer
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import time
import psutil
import os  # Add this line to import the os module

# Try to load BERT tokenizer and model from local directory, fall back to pre-trained if not found
try:
    bert_tokenizer = BertTokenizer.from_pretrained('./results/bert')
    bert_model = BertForNextSentencePrediction.from_pretrained('./results/bert')
    print("Loaded BERT model and tokenizer from local directory.")
except OSError:
    print("Local BERT model not found. Loading pre-trained model.")
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

# Load ALBERT and RoBERTa models for sentence generation
albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
albert_model = AlbertForMaskedLM.from_pretrained('albert-base-v2')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForMaskedLM.from_pretrained('roberta-base')

# Load the WikiText dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
val_texts = dataset['validation']['text']

def generate_sentence(model, tokenizer, prompt, max_length=32):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    
    # Mask a random token
    rand_index = torch.randint(0, input_ids.shape[1], (1,))
    input_ids[0, rand_index] = tokenizer.mask_token_id
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        predictions = outputs.logits
    
    masked_token_predictions = predictions[0, rand_index]
    top_token = torch.argmax(masked_token_predictions, dim=-1)
    
    input_ids[0, rand_index] = top_token
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def prepare_sentence_pairs(texts, model, tokenizer, max_length=32):
    sentence_pairs = []
    for text in texts:
        sentences = text.split(". ")
        for i in range(len(sentences) - 1):
            sentence_a = sentences[i]
            sentence_b = generate_sentence(model, tokenizer, sentence_a, max_length)
            encoded = bert_tokenizer(sentence_a, sentence_b,
                                     padding='max_length',
                                     truncation=True,
                                     max_length=max_length,
                                     return_tensors='pt')
            encoded['labels'] = torch.tensor([0])
            sentence_pairs.append({k: v.squeeze(0) for k, v in encoded.items()})
            
            if i + 2 < len(sentences):
                sentence_c = generate_sentence(model, tokenizer, sentences[i+2], max_length)
                encoded = bert_tokenizer(sentence_a, sentence_c,
                                         padding='max_length',
                                         truncation=True,
                                         max_length=max_length,
                                         return_tensors='pt')
                encoded['labels'] = torch.tensor([1])
                sentence_pairs.append({k: v.squeeze(0) for k, v in encoded.items()})
    return Dataset.from_list(sentence_pairs)

# Prepare datasets for validation
val_data = val_texts[:100]
albert_dataset = prepare_sentence_pairs(val_data, albert_model, albert_tokenizer)
roberta_dataset = prepare_sentence_pairs(val_data, roberta_model, roberta_tokenizer)

def evaluate_model(model, dataset):
    model.eval()
    eval_results = []
    inference_times = []
    for val_instance in dataset:
        start_time = time.time()
        with torch.no_grad():
            # Convert inputs to tensors if they're not already
            inputs = {k: torch.tensor(v).unsqueeze(0) if isinstance(v, list) else v.unsqueeze(0) 
                      for k, v in val_instance.items() if k != 'labels'}
            labels = torch.tensor(val_instance['labels']).unsqueeze(0)
            
            outputs = model(**inputs, labels=labels)
        
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=-1)
        eval_results.append((predicted_label.item(), val_instance['labels']))

    predictions, labels = zip(*eval_results)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary', zero_division=0)
    recall = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='weighted')
    auc_roc = roc_auc_score(labels, predictions)
    avg_inference_time = sum(inference_times) / len(inference_times)
    throughput = 1 / avg_inference_time

    return {
        "performance_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc
        },
        "inference_metrics": {
            "avg_inference_time": avg_inference_time,
            "throughput": throughput
        }
    }

# Evaluate the model on ALBERT and RoBERTa generated sentences
albert_results = evaluate_model(bert_model, albert_dataset)
roberta_results = evaluate_model(bert_model, roberta_dataset)

# Get model sizes
bert_model_size = sum(p.numel() * p.element_size() for p in bert_model.parameters()) / (1024 * 1024)
albert_model_size = sum(p.numel() * p.element_size() for p in albert_model.parameters()) / (1024 * 1024)
roberta_model_size = sum(p.numel() * p.element_size() for p in roberta_model.parameters()) / (1024 * 1024)

# Get memory usage
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / (1024 * 1024)

# Print results
print("ALBERT Generated Sentences Results:")
print(albert_results)
print("\nRoBERTa Generated Sentences Results:")
print(roberta_results)
print("\nResource Utilization:")
print(f"BERT Model Size: {bert_model_size:.2f} MB")
print(f"ALBERT Model Size: {albert_model_size:.2f} MB")
print(f"RoBERTa Model Size: {roberta_model_size:.2f} MB")
print(f"Memory Usage: {memory_usage:.2f} MB")