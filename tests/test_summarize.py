"""Test file for summarize.py rudimentary functions.

clean_text()
calculate_tf()
calculate_idf()
score_sentences()

Completed by Shaun W. for Lab 8."""

import summarize as sum
from summarize import clean_text

test_switch = True

sample_1 = ["Hello, this is a sentence!",
            "Do woodchucks chuck wood?"]
sample_2 = ["How is the weather?",
            "Make sure it is sunny!"]

sample_tf1 = sum.calculate_tf(clean_text(sample_1))
sample_tf2 = sum.calculate_tf(clean_text(sample_2))
sample_idf1 = sum.calculate_idf(clean_text(sample_1))
sample_idf2 = sum.calculate_idf(clean_text(sample_2))


def test_clean():
    assert sum.clean_text(sample_1) == [
                ["hello", "this", "is", "a", "sentence"],
                ["do", "woodchucks", "chuck", "wood"]
                ]
    assert sum.clean_text(sample_2) == [
                ['how', 'is', 'the', 'weather'],
                ['make', 'sure', 'it', 'is', 'sunny']
                ]


def test_tf():
    assert sum.calculate_tf(sum.clean_text(sample_1)) == [
              {'hello':0.2, 'this':0.2, 'is':0.2, 'a':0.2, 'sentence':0.2},
              {'do':0.25, 'woodchucks':0.25, 'chuck':0.25, 'wood':0.25}
             ]
    assert sum.calculate_tf(sum.clean_text(sample_2)) == [
              {'how':0.25, 'is':0.25, 'the':0.25, 'weather':0.25},
              {'make':0.2, 'sure':0.2, 'it':0.2, 'is':0.2, 'sunny':0.2}
             ]


def test_idf():
    assert sum.calculate_idf(clean_text(sample_1)) == {
        'hello':0.5, 'this':0.5, 'is':0.5, 'a':0.5, 'sentence':0.5,
        'do':0.5, 'woodchucks':0.5, 'chuck':0.5, 'wood':0.5,
    }
    assert sum.calculate_idf(clean_text(sample_2)) == {
        'how':0.5, 'is':1, 'the':0.5, 'weather':0.5,
        'make':0.5, 'sure':0.5, 'it':0.5, 'sunny':0.5
    }


def test_score():
    assert sum.score_sentences(sample_tf1, sample_idf1, clean_text(sample_1)) == [0.5,0.5]
    assert sum.score_sentences(sample_tf2, sample_idf2, clean_text(sample_2)) == [0.625,0.6]


if test_switch:
    test_clean()
    test_tf()
    test_idf()
    test_score()