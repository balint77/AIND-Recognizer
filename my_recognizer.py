import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # TODO implement the recognizer
    test_sequences = list(test_set.get_all_Xlengths().values())
    probabilities = [{}] * len(test_sequences)
    guesses = [None] * len(test_sequences)
    for test_word_idx, (test_X, test_lengths) in enumerate(test_sequences):
        best_logL = float("-inf")
        for word, model in models.items():
            try:
                logL = float("-inf") if model is None else model.score(test_X, test_lengths)
            except ValueError:
                logL = float("-inf")
            probabilities[test_word_idx][word] = logL
            if logL > best_logL:
                best_logL = logL
                best_word = word
        guesses[test_word_idx] = best_word
    return probabilities, guesses
