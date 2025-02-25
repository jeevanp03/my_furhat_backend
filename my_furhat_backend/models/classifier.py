from transformers import pipeline
class TextClassifier:
    def __init__(self, model_id = "facebook/bart-large-mnli"):
        self.classifier = pipeline("zero-shot-classification", model=model_id)
    
    def classify(self, text: str, labels: list) -> dict:
        scores = self.classifier(text, labels, multi_label=True)
        scores_with_labels = dict(zip(scores["labels"], scores["scores"]))
        scores_with_labels_ascending = dict(sorted(scores_with_labels.items(), key=lambda item: item[1], reverse=True))
        return scores_with_labels_ascending

# text_classifier = TextClassifier()

# __all__ = ["TextClassifier", "text_classifier"]