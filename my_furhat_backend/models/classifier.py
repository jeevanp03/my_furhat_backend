from transformers import pipeline

class TextClassifier:
    """
    A simple text classifier using a zero-shot classification pipeline.

    This class leverages Hugging Face's transformers library to perform zero-shot 
    classification, which allows classification without task-specific training data.
    """

    def __init__(self, model_id="facebook/bart-large-mnli"):
        """
        Initialize the TextClassifier with a specified model.

        Parameters:
            model_id (str): The Hugging Face model identifier to use for zero-shot classification.
                            Default is "facebook/bart-large-mnli".
        """
        # Create a zero-shot classification pipeline using the specified model.
        self.classifier = pipeline("zero-shot-classification", model=model_id)
    
    def classify(self, text: str, labels: list) -> dict:
        """
        Classify the given text into one or more labels using zero-shot classification.

        Parameters:
            text (str): The text to classify.
            labels (list): A list of candidate labels for classification.

        Returns:
            dict: A dictionary mapping each label to its corresponding score, sorted in descending order.
        """
        # Perform zero-shot classification on the input text with the given labels,
        # allowing multiple labels to be assigned (multi_label=True).
        scores = self.classifier(text, labels, multi_label=True)
        
        # Create a dictionary mapping labels to their scores.
        scores_with_labels = dict(zip(scores["labels"], scores["scores"]))
        
        # Sort the dictionary by scores in descending order.
        scores_with_labels_ascending = dict(
            sorted(scores_with_labels.items(), key=lambda item: item[1], reverse=True)
        )
        
        return scores_with_labels_ascending
