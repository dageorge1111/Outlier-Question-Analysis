import os
import json
import torch
from transformers import BertModel, BertTokenizer
from sklearn.ensemble import IsolationForest
import numpy as np
from dotenv import load_dotenv

def load_bert_model():
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased')
  model.eval()
  return tokenizer, model

def preprocess_text(tokenizer, text, max_length=512):
  inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
  return inputs

def extract_text_features(text_list, tokenizer, model):
  features = []
  with torch.no_grad():
    for text in text_list:
      inputs = preprocess_text(tokenizer, text)
      outputs = model(**inputs)
      cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
      features.append(cls_embedding)
  return np.vstack(features)

def run_isolation_forest(features, contamination=0.1):
  isolation_forest = IsolationForest(contamination=contamination, random_state=42)
  isolation_forest.fit(features)
  outlier_predictions = isolation_forest.predict(features)
  outliers = np.where(outlier_predictions == -1)[0]
  return outliers, outlier_predictions

def main():
  tokenizer, bert_model = load_bert_model()
  text_data = ["This is an example text.", "Another sample input.", "Potentially anomalous data."]
  
  features = extract_text_features(text_data, tokenizer, bert_model)
  outlier_indices, predictions = run_isolation_forest(features)
  print(outlier_indices)

if __name__ == "__main__":
  main()
