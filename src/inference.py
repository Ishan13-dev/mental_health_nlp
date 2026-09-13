import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

class MentalHealthClassifier:
    def __init__(self, model_path="Ishan13-dev/mental-health-nlp"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs = probs.cpu().numpy()[0]

        # Map ids to labels using the model's config
        id2label = self.model.config.id2label
        
        # Sort predictions by probability
        sorted_indices = probs.argsort()[::-1]
        results = [
            {"label": id2label[idx], "confidence": float(probs[idx])}
            for idx in sorted_indices
        ]
        
        return results