from typing import List, Dict
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    # Open the file, go line by line, separate sentence and label, tokenize the sentence and create SentimentExample object
    with open(infile, 'r') as file:
        lines: List[str] = file.readlines()
    sentence_value: List[List[str, int]] = [line.rsplit(sep="\t", maxsplit=1) for line in lines]  # last element separated by "\t" is the value
    examples: List[SentimentExample] = [SentimentExample(tokenize(sentence.lower()), value) for sentence, value in sentence_value]
    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    # Count unique words in all the examples from the training set
    unique_words = []
    for sent_example in examples:
        unique_words += [word for word in sent_example.words if word not in unique_words]
    
    vocab: Dict[str, int] = {}
    for i in range(len(unique_words)):
        vocab[unique_words[i]] = i
    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    if binary:
        bow: torch.Tensor = torch.tensor([1 if word in text else 0 for word in list(vocab)])
    else:
        bow: torch.Tensor = torch.zeros((len(vocab)))
        for word, idx in vocab.items():
            bow[idx] += 1 if word in text else 0
    return bow
