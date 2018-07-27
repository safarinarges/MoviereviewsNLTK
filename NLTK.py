
 
#voting system to choose which classifier is more relibale and confidence paramater 
    
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
#to choose who got the most votes
from statistics import mode

#build a new classifier
#pass the list of classifiers through the VoteClassfier (*)
class VoteClassifier (ClassifierI):
    def __init__ (self, *classifiers):
        self._classifiers = classifiers
        
    def classify (self, features):
        votes =[]
        for c in self._classifiers:
            v = c.classify (features)
            votes.append(v)
        return mode (votes)
    
    def confiednce (self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
#below counts how many occurrences of that most popular vote were in that list
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
        
        
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist (all_words)
word_features = list (all_words.keys())[:3000]

def find_features(document):
    words = set (document)
    features = {}
    for w in word_features:
        features [w] = (w in words)
        
    return features


featuresets = [(find_features(rev),category)for (rev, category) in documents]
        
training_set = featuresets[:1900]
testing_set = featuresets [1900:]

#open the classifier; rb to read and it was saved in bytes in prev lesson
classifier_f = open ("naivebayes.pickle", "rb")
classifier = pickle.load (classifier_f)
classifier_f.close()
print("original Naive Bayes Algo Accuracy Percentage:", (nltk.classify.accuracy(classifier, testing_set))*100)
#classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier (MultinomialNB())
MNB_classifier.train (training_set)
print("MNB_classifier Accuracy Percentage:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BNB_classifier = SklearnClassifier (BernoulliNB())
BNB_classifier.train (training_set)
print("BNB_classifier Accuracy Percentage:", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier (LogisticRegression())
LogisticRegression_classifier.train (training_set)
print("LogisticRegression_classifier Accuracy Percentage:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier (SGDClassifier())
SGDClassifier_classifier.train (training_set)
print("SGDClassifier_classifier Accuracy Percentage:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier (SVC())
SVC_classifier.train (training_set)
print("SVC_classifier Accuracy Percentage:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier (LinearSVC())
LinearSVC_classifier.train (training_set)
print("LinearSVC_classifier Accuracy Percentage:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier (NuSVC())
NuSVC_classifier.train (training_set)
print("NuSVC_classifier Accuracy Percentage:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)



voted_classfier = VoteClassifier(classifier, 
                                 SVC_classifier,
                                 NuSVC_classifier, 
                                 LinearSVC_classifier, 
                                 SGDClassifier_classifier, 
                                 MNB_classifier,BNB_classifier, 
                                 LogisticRegression_classifier)

#general percent based on the entire testing sample and the second print is just one example
print("voted_classfier Accuracy Percentage:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

#testin_set is a list of features
print("Classification:", voted_classfier.classify(testing_set[0][0]), "confidence %:", voted_classfier.confiednce(testing_set[0][0])*100)
print("Classification:", voted_classfier.classify(testing_set[1][0]), "confidence %:", voted_classfier.confiednce(testing_set[1][0])*100)
print("Classification:", voted_classfier.classify(testing_set[2][0]), "confidence %:", voted_classfier.confiednce(testing_set[2][0])*100)
print("Classification:", voted_classfier.classify(testing_set[3][0]), "confidence %:", voted_classfier.confiednce(testing_set[3][0])*100)
print("Classification:", voted_classfier.classify(testing_set[4][0]), "confidence %:", voted_classfier.confiednce(testing_set[4][0])*100)
print("Classification:", voted_classfier.classify(testing_set[5][0]), "confidence %:", voted_classfier.confiednce(testing_set[5][0])*100)








