import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import torch

df = pd.read_csv('Tweets.csv')

# Keep only the necessary columns
df = df[['text', 'airline_sentiment']]

# Convert sentiment labels to numerical values (0 for negative, 1 for neutral, 2 for positive)
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['airline_sentiment'].map(sentiment_mapping)

# Select a subset of the dataset (for simplicity and faster training)
df = df.sample(frac=0.1, random_state=42)

# Split the dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Explicitly download tokenizer and model with cache_dir
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./cache')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', cache_dir='./cache', num_labels=3)

# Download tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Tokenize the texts
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)

# Prepare a PyTorch Dataset
class AirlineSentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Prepare the datasets
train_dataset = AirlineSentimentDataset(train_encodings, train_labels)
test_dataset = AirlineSentimentDataset(test_encodings, test_labels)

# Set training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    output_dir='./results',
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=500,
    evaluation_strategy="steps"
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Fine-tune the model
trainer.train()

# Evaluate the model
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = test_labels.tolist()

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")
