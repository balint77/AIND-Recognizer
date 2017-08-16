import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose
        if self.verbose:
            print("X {}, lengths {}".format(self.X, self.lengths))

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on BIC scores
        #max_components = min(len(min(self.sequences, key=len)),self.max_n_components)
        if self.verbose:
            print("BIC selector for {} has {} sequences".format(self.this_word, len(self.sequences)))
        best_bic = float("inf")
        best_model = None
        for num_components in range(self.min_n_components, self.max_n_components + 1):
            param_count = num_components * num_components + 2 * num_components * len(self.X[0]) - 1
            try:
                model = GaussianHMM(n_components=num_components, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                #if best_model is None:
                #    best_model = model #have at least some result
                if self.verbose:
                    print("model done")
                logL = model.score(self.X, self.lengths)
                if self.verbose:
                    print("scoring done")
                bic = -2 * logL + param_count * math.log(len(self.X))
            except ValueError:
                logL = float("-inf")
                bic = float("inf")
            if self.verbose:
                print("{} state count has {} logL and {} BIC".format(num_components, logL, bic))
            if bic < best_bic:
                best_bic = bic
                best_model = model
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        if self.verbose:
            print("DIC selector for {} has {} sequences".format(self.this_word, len(self.sequences)))
        best_dic = float("-inf")
        best_model = None
        for num_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components=num_components, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                otherWordCount = 0
                otherLogL = 0
                for other_word in self.words:
                    try:
                        if other_word == self.this_word:
                            logL = model.score(self.X, self.lengths)
                        else:
                            otherX, otherLengths = self.hwords[other_word]
                            otherLogL = otherLogL + model.score(otherX, otherLengths)
                            otherWordCount += 1
                    except ValueError:
                        if other_word == self.this_word:
                            logL = float("-inf")
                            break
                if otherWordCount == 0:
                    otherLogL = float("inf")
                else:
                    otherLogL = otherLogL / otherWordCount
                dic = logL - otherLogL
                if self.verbose:
                    print("{} state count has {} logL and {} DIC from {} words".format(num_components, logL, dic, otherWordCount))
                if dic > best_dic:
                    best_dic = dic
                    best_model = model
            except ValueError:
                pass
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        #max_components = min(len(min(self.sequences, key=len)),self.max_n_components)
        if self.verbose:
            print("CV selector for {} has {} sequences".format(self.this_word, len(self.sequences)))
        best_logL = float("-inf")
        best_model = None
        for num_components in range(self.min_n_components, self.max_n_components + 1):
            if len(self.sequences) == 1:
                runCount = 1
                try:
                    model = GaussianHMM(n_components=num_components, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    logL = model.score(self.X, self.lengths)
                except ValueError:
                    logL = float("-inf")
            else:
                #split_method = KFold(n_splits=min(math.ceil(math.sqrt(len(self.sequences))), 10))
                split_method = KFold(n_splits=min(len(self.sequences), 10))
                runCount = 0
                logL = 0
                for train_idx, test_idx in split_method.split(self.sequences):
                    try:
                        # print("Train fold indices:{} Test fold indices:{} for {}".format(cv_train_idx, cv_test_idx, self.this_word))
                        train_X, train_lengths = combine_sequences(train_idx, self.sequences)
                        model = GaussianHMM(n_components=num_components, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
                        test_X, test_lengths = combine_sequences(test_idx, self.sequences)
                        logL = logL + model.score(test_X, test_lengths)
                        runCount += 1
                    except ValueError:
                        #print("Fail for {} state count".format(num_components))
                        pass
                if runCount == 0:
                    logL = float("-inf")
                else:
                    logL = logL / runCount
            if self.verbose:
                print("{} state count has {} with {} loops".format(num_components, logL, runCount))
            if logL > best_logL:
                best_logL = logL
                best_model = model
        return best_model
