# Use a pipeline as a high-level helper
from transformers import pipeline
import torch
import gradio as gr

pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")


# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

# Define a function for sentiment analysis
def predict_sentiment(text):
    # Tokenize the input text and prepare it to be used by the model
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted probabilities and convert them to percentages
    probabilities = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
    positive_percent = probabilities[2] * 100
    negative_percent = probabilities[0] * 100
    neutral_percent = probabilities[1] * 100

    # Construct the result dictionary
    result = {
        "Positive": round(positive_percent, 2),
        "Negative": round(negative_percent, 2),
        "Neutral": round(neutral_percent, 2)
    }

    return result
