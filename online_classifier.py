from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import os
import numpy as np
import cPickle as pickle

class OnlineClassifier:

    def __init__(self):
        self.supported_labels = ['pos', 'neg']
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.supported_labels)

        # seed
        seed_examples = ['I like pie', 'i hate chicken']
        seed_labels = ['pos', 'neg']

        self.vectorizer = CountVectorizer(min_df=1, binary=True) #we only use binary features

        seed_X = self.vectorizer.fit_transform(seed_examples)
        seed_y = self.label_encoder.transform(seed_labels)

        self.model = SGDClassifier(loss='log')
        self.model.partial_fit(seed_X, seed_y, classes=seed_y)

        self.vocabulary = dict([(name, idx) for idx, name in enumerate(self.vectorizer.get_feature_names())])

    def save(self, path=""):
        root = os.path.abspath(path) if path else os.getcwd()

        f = os.path.join(root, 'online_classifier.pickle')

        data = {
        'model': self.model,
        'label_encoder': self.label_encoder,
        'supported_labels': self.supported_labels,
        'vocabulary': self.vocabulary,
        'vectorizer': self.vectorizer,
        }

        pickle.dump(data, open(f, 'wb'))

    def load(self, path=""):
        root = os.path.abspath(path) if path else os.getcwd()

        f = os.path.join(root, 'online_classifier.pickle')

        data = pickle.load( open( f, "rb" ) )

        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.supported_labels = data['supported_labels']
        self.vocabulary = data['vocabulary']
        self.vectorizer = data['vectorizer']

    def train(self, new_examples, new_labels):

        new_vectorizer = CountVectorizer(min_df=1, binary=True)
        new_vectorizer.fit(new_examples)
        new_words = new_vectorizer.get_feature_names()

        unknown_words = list(set(new_words) - set(self.vocabulary.keys()))

        for w in unknown_words:
            self.vocabulary[w] = len(self.vocabulary)

        # enlarge our vocabulary and our weight vector
        new_weights = np.zeros(shape=(self.model.coef_.shape[0], len(unknown_words)))
        self.model.coef_ = np.concatenate((self.model.coef_, new_weights), axis=1)

        self.vectorizer = CountVectorizer(min_df=0, binary=True, vocabulary = self.vocabulary)
        new_X = self.vectorizer.fit_transform(new_examples)
        new_y = self.label_encoder.transform(new_labels)

        self.model.partial_fit(new_X, new_y)

    def predict(self, new_examples):
        new_X = self.vectorizer.transform(new_examples)
        new_y = [ self.supported_labels[y] for y in self.model.predict(new_X).tolist()]
        new_proba = self.model.predict_proba(new_X).tolist()[0]
        return new_y, new_proba

    def get_coefficients(self):
        coefficients = {}
        for word, index in self.vocabulary.items():
            coefficients[word] = self.model.coef_.tolist()[0][index]
        return coefficients

    def get_max_coefficient(self):
        coefs = self.model.coef_.tolist()[0]
        max_coef = max(coefs)
        min_coef = min(coefs)
        return max_coef if max_coef > abs(min_coef) else abs(min_coef)

    def get_coefficients_for(self, sentence):
        answer = []
        for w in sentence.split():
            coefficient = self.model.coef_.tolist()[0][ self.vocabulary[w] ] if w in self.vocabulary.keys() else 0
            answer += [(w, coefficient)]
        return answer
