import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # Estimate class priors and conditional probabilities of the bag of words 
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = features.shape[1] # Shape of the probability tensors, useful for predictions and conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # Count number of samples for each output class and divide by total of samples
        label_count: Dict[int,int] = Counter(labels.tolist())  # Dict[label, times_it_appears_in_sample]
        total: int = sum(label_count.values())
        class_priors: Dict[int, torch.Tensor] = {label: torch.tensor(count / total, dtype=torch.float32) for label, count in label_count.items()}
        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        # Bayes Theorem: P(w|c) = (P(w,c)*P(w)) / P(c)
        # self.class_priors = P(c)

        # Estimate conditional probabilities for the words in features and apply smoothing
        label_count: Dict[int,int] = Counter(labels)  # {label: num_ocurrencies}
        total: int = sum(list(label_count.values()))  # total_num_ocurrencies

        # Laplace smoothing
        for label in list(label_count):
            label_count[label] += delta
        total += (delta * self.vocab_size)

        # Probability of the labels (P(c))
        P_label: Dict[int, float] = {}
        for label, frecuency in label_count.items():
            P_label[label] = frecuency / total
        
        # Joint probability (P(w,c))
        # for label in list(label_count):
        #     for example, label_ex in labels:
        class_word_counts: Dict[int, torch.Tensor] = {}
        # word_counts: torch.tensor = torch.zeros((len(list(label_count)), self.vocab_size))  # word_counts.shape = [num_classes, len_vocab]
        # total_words = word_counts.sum(dim=1)
        for label in list(label_count):
            class_indices = (labels == label).nonzero(as_tuple=True)[0]  # Get indices for this class
            class_features = features[class_indices]  # Filter features for this class
            word_counts = class_features.sum(dim=0)  # Sum occurrences of each word
            total_words = word_counts.sum().item()  # Total words in this class
        
        # Apply Laplace smoothing
            smoothed_probs = (word_counts + delta) / (total_words + delta * self.vocab_size)
            class_word_counts[label.item()] = smoothed_probs
        
        
        # class_word_counts: Dict[int, torch.Tensor] = {idx_word: torch.dot(features)}
        
        self.conditional_probabilities = class_word_counts
        return class_word_counts

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        # Calculate posterior based on priors and conditional probabilities of the words
        log_posts: list[torch.Tensor] = []
        for label in list(self.class_priors):
            log_prior = torch.log(self.class_priors[label])
            log_likelihood = (feature * torch.log(self.conditional_probabilities[label])).sum()
            log_posts.append(log_prior + log_likelihood)
        log_posteriors: torch.Tensor = torch.stack(log_posts)
        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        # Calculate log posteriors and obtain the class of maximum likelihood 
        log_posteriors: torch.Tensor = self.estimate_class_posteriors(feature)
        pred: int = torch.argmax(log_posteriors).item()
        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # Calculate log posteriors and transform them to probabilities (softmax)
        log_posteriors: torch.Tensor = self.estimate_class_posteriors(feature)
        probs: torch.Tensor = torch.softmax(log_posteriors, 0)
        return probs
