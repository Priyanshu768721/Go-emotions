from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import numpy as np

# Load model and tokenizer
model_path = "roberta-goemotions-model"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

# Move model to available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Emotion labels
label_names = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

for i in range(5):
    text = input("Enetr ur emotions : ")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()  # Move to CPU for numpy

    sorted_indices = np.argsort(-probs)
    top_k=5

    for idx in sorted_indices[:top_k]:
        print(f"**{label_names[idx].capitalize()}**")
        print(float(probs[idx]))
