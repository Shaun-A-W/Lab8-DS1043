#!/usr/bin/env python
"""summarize

Summarize a document using extractive text summarization via tf-idf.

Usage:
  summarize [-o <file> | --output=<file>] [<input-file>]
  summarize (-h | --help)

Options:
  -h --help            Show this screen.
  -o --output=<file>   Write output to file instead of stdout.
"""
from docopt import docopt
import nltk
import sys
from collections import defaultdict
from collections.abc import Callable
from typing import TextIO


# sample = ["Hello, this is a sentence!", "Hmm... How do I talk about a woodchuck?",
#         "If a woodchuck could chuck wood, how much wood could he chuck?"]


def load_document(textfile: TextIO) -> list[str]:
    """Reads a text file and returns a list of sentences"""
    text = [line.strip() for line in textfile.readlines()]
    text = nltk.sent_tokenize(' '.join(text))
    return text


def clean_text(text: list[str]) -> list[list[str]]:
    """Transform text into a list of terms for each sentence"""
    sentences: list[list[str]] = []
    for line in text:
        # Creates list of terms/words
        sentence = [word.casefold()
                    for word in nltk.word_tokenize(line)]
        # Removes terms that are not alphanumeric.
        for word in sentence:
            if word.isalnum():
                continue
            else: sentence.remove(word)
        # If the resulting list/sentence is non-empty, add to final list
        if len(sentence) > 0:
            sentences.append(sentence)
    return sentences

                

def calculate_tf(sentences: list[list[str]]) -> list[dict]:
    """Calculate Term Frequency for each sentence of the document
    Returns a table whose keys are the indices of sentences of the text
    and values are dictionaries of terms and their tf values."""
    matrix: list[dict] = []

    # Adds dictionary of words and the repetition count
    # for each sentence
    for sentence in sentences:
        temp_dict = {}
        for word in sentence:
            if word not in temp_dict:
                temp_dict[word] = 1
            else:
                temp_dict[word] += 1
        matrix.append(temp_dict)

    # Goes through previous dictionaries
    # Adjusts value weights into total count / length of sentence
    for entry in matrix:
        term_count = len(entry)
        for term in entry:
            entry[term] = entry[term] / term_count

    return matrix


def calculate_idf(sentences: list[list[str]]) -> dict[str, float]:
    """Calculate the Inverse `Document'(Sentence) Frequency of each term.
    Returns a table of terms and their idf values."""
    matrix: dict[str, float] = defaultdict(float)
    doc_len = len(sentences)

    # Goes through document and finds relative frequency
    # across whole corpus
    for sentence in sentences:
        for word in sentence:
            if word not in matrix:
                matrix[word] = 1/doc_len
            else:
                matrix[word] += 1/doc_len

    return matrix


def score_sentences(tf_matrix: list[dict], idf_matrix: dict[str, float], sentences: list[list[str]]) -> list[float]:
    """Score each sentence for importance based on the terms it contains.
    Assumes that there are no empty sentences.
    Returns a table whose keys are the indices of sentences of the text
    and values are the sum of tf-idf scores of each word in the sentence"""
    scores: list[float] = []

    # Create indices for sentences
    for n, sentence in enumerate(sentences):
        sent_score = 0
        # Goes through each word and finding its value
        for word in sentence:
            temp_score = 0
            temp_score += tf_matrix[n][word] * idf_matrix[word]
        # Adds word value to sentence value
            sent_score += temp_score
        # Adds sentence total value to final list
        scores.append(sent_score)

    return scores


def threshold_inclusion(text: list[str], scores: list[float], threshold=1):
    """Use a multiple of the average tf-idf document score as a threshold for inclusion in summary"""
    avg_score = sum(scores) / len(scores)
    summary = []
    for index, score in enumerate(scores):
        if score >= threshold * avg_score:
            summary += [text[index]]
    return summary
    

def summarize(text: list[str], inclusion: Callable) -> str:
    """Summarizes a given text using tf-idf and a given inclusion function."""
    sentences = clean_text(text)
    tf_matrix = calculate_tf(sentences)
    idf_matrix = calculate_idf(sentences)
    scores = score_sentences(tf_matrix, idf_matrix, sentences)
    summary = inclusion(text, scores)
    return ' '.join(summary) + '\n'


if __name__ == '__main__':
    arguments = docopt(__doc__)
    if arguments['<input-file>']:
        with open(arguments['<input-file>'], 'r', encoding='utf-8') as infile:
            document = load_document(infile)
    else:
        document = load_document(sys.stdin)

    # Threshold value may need adjustment. It might be appropriate to expand this
    # to allow inclusion function and inclusion criteria to be specified as
    # commandline options
    func = lambda text, scores: threshold_inclusion(text, scores, threshold=1)

    if arguments['--output']:
        with open(arguments['--output'], 'w', encoding='utf-8') as outfile:
            outfile.write(summarize(document, func))
    else:
        sys.stdout.write(summarize(document, func))