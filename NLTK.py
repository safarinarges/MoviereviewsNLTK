

# =============================================================================
# #Part One
# =============================================================================
"""
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
example="Tens of thousands of people have joined nationwide protests across the US over the Trump administration's hardline immigration policies. Major protests took place in Washington DC, New York"
print(sent_tokenize (example)) 
print(word_tokenize(example))
for i in word_tokenize(example):
    print(i)
"""
# =============================================================================
# Part Two: Stop Words
# =============================================================================
"""
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


example_sentence="This is an example showing of stopword filteration."
stop_words=set(stopwords.words("english"))
#print(stop_words)

filtered_sentence=[]
for w in word_tokenize (example_sentence):
    if w not in stop_words:
       filtered_sentence.append(w)
print(filtered_sentence)
"""
# =============================================================================
# #Part Three: Stemming words with NLTK
# =============================================================================
"""
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

ps= PorterStemmer()

example_words=["python", "Pythoner","Pythoning","Pythoned"]

for w in example_words:
    print(ps.stem(w))
    
second_example="It is important to learn python while growing in your comapny which is a pythoning company."

for w in word_tokenize (second_example):
    print(ps.stem(w))
#Combine stem and stopwords
stop_words=set(stopwords.words("english"))
filter_sentence=[]
for w in word_tokenize(second_example):
    if ps.stem(w) not in stop_words:
        filter_sentence.append(w)
print(filter_sentence)
"""
# =============================================================================
# Part Four: Part of Speech Tagging
# =============================================================================

"""
import nltk
from nltk.corpus import state_union
#state_union are different presidents speech over the last 70 years
from nltk.tokenize import PunktSentenceTokenizer
#PunktSentenceTokenizer is unsupervised machine learning sentence toeknizer

train_text=state_union.raw("2006-GWBush.txt")
sample_text=state_union.raw("2006-GWBush.txt")

custome_sent_tokenizer=PunktSentenceTokenizer(train_text)
tokenized=custome_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words=nltk.word_tokenize(i)
            tagged=nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))
#    except FileNotFoundError:
#        print("Sorry the file does not exsist")
            

process_content()
"""
#Q: train PunktSentenceTokenizer to acccept the adjectives and the words that we define to recognise if it is a protest or not?


# =============================================================================
# Regular Expressions
# =============================================================================
"""
import re
exampleString= '''
Jessica is 15 years old and Peter has the same age.
Edward is 97 and Narges is 77'''
ages=re.findall(r'\d {1,3}', exampleString)
names = re.findall(r'[A-Z][a-z]*', exampleString)
print(ages)
print(names)
"""

# =============================================================================
# Part Five: Chunking 
# =============================================================================
"""
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text=state_union.raw("2005-GWBush.txt")
sample_text=state_union.raw("2006-GWBush.txt")

custome_sent_tokenizer=PunktSentenceTokenizer(train_text)
tokenized=custome_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words=nltk.word_tokenize(i)
            tagged=nltk.pos_tag(words)
#           print(tagged)
    
#chunkGram is we are looking for any form of an adverb (RB) and we are looking for 0 or mor eo fthis (*), VB is verb; NNp is proper noun
#there is """"""after r and the } in the next line
            chunkGram = rChunk: { <RB.?>*<VB.?>*<NNP><NN>?}
                        chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            chunked.draw()

            
    except Exception as e:
        print(str(e))
        
process_content()
"""
# =============================================================================
# Chapter six: Chinking
# =============================================================================
"""
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text=state_union.raw("2005-GWBush.txt")
sample_text=state_union.raw("2006-GWBush.txt")

custome_sent_tokenizer=PunktSentenceTokenizer(train_text)
tokenized=custome_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words=nltk.word_tokenize(i)
            tagged=nltk.pos_tag(words)
#we chunked everything, then we chink }<>{ stuff we dont want to chunk
            chunkGram = r Chunk: { <.*>+}
                                        }<VB.?|IN|DT>+{
            
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            chunked.draw()

            
    except Exception as e:
        print(str(e))
        
process_content()
"""
# =============================================================================
# Chapter Seven: Named Entity Recognition
# =============================================================================
"""
NE Type and Examples
ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, Midlothian
"""
"""
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text=state_union.raw("2005-GWBush.txt")
sample_text=state_union.raw("2006-GWBush.txt")

custome_sent_tokenizer=PunktSentenceTokenizer(train_text)
tokenized=custome_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words=nltk.word_tokenize(i)
            tagged=nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=True)
                        
            namedEnt.draw()

            
    except Exception as e:
        print(str(e))
        
process_content()
"""

# =============================================================================
# Lesson Eight: Lemmatizing
# =============================================================================
"""
#think lemmetizing is more practical than stemming
from nltk.stem import WordNetLemmatizer

lematizer = WordNetLemmatizer()

print(lematizer.lemmatize("better", pos = "a" ))
print(lematizer.lemmatize("running", pos = "v" ))
print(lematizer.lemmatize("nice", pos = "a" ))

"""

# =============================================================================
# Lesson Nine: Corpora with NLTK
# =============================================================================

#import nltk
#print(nltk.__file__)
"""
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize
sample = gutenberg.raw("bible-Kjv.txt")

tok = sent_tokenize (sample)
print(tok[1:5])

"""
# =============================================================================
# Lesson Ten: WordNet
# =============================================================================
"""
from nltk.corpus import wordnet

syns = wordnet.synsets("program")

#Synset
print(syns[0].name())

#Just the word
print(syns[0].lemmas()[0].name())

#Definition
print(syns[0].definition())

#examples
print(syns[0].examples())

synonyms =[]
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
#        print("l:",l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms ()[0].name())
print(set(synonyms))
print(set(antonyms))

#Semantic Similarity: compare the similarity between the first synonym of these two words to compare the symantic similarity
 #wup: Wu and Palmer method for semantic similarity
 
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset ('boat.n.01')
print(w1.wup_similarity(w2))


w1 = wordnet.synset('protest.n.01')
w2 = wordnet.synset ('object.n.01')
print(w1.wup_similarity(w2))
 
"""
# =============================================================================
# Lesson 11: Text Classification
# =============================================================================
"""
#random is used to shuffle up the dataset that we have becasue the file we have is in highly ordered the first 5000 reviews are negative and the second half are positive        
import nltk
import random
from nltk.corpus import movie_reviews

#documents is to train and test set but we compile the massive list of all-words
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]


#the bovae code for fillinf the document is simlar to:
#    documents = []
#    for category in movie_reviews.categories():
#        for fileid in movie_reviews.fileids(categoriy):
#            documents.append(list(movie_reviews.words(fileid)), category)
#

random.shuffle(documents)
#print(documents[1])

#lower is because we make sure everything is normalised, so no lower/upper casing

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

#covert all words to nltk frequency distribution to find the most common words
all_words = nltk.FreqDist (all_words)

print(all_words.most_common(15))

print(all_words["stupid"])
"""
# =============================================================================
# Lesson Twelve: Converting words to Features with NLTK
# =============================================================================
"""
import nltk
import random
from nltk.corpus import movie_reviews

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

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev),category)for (rev, category) in documents]
        
"""
# =============================================================================
# Chapter 13: Naive Bayes classifier
# =============================================================================
"""

import nltk
import random
from nltk.corpus import movie_reviews

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

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Naive Bayes Algo Accuracy Percentage:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)
"""

# =============================================================================
# Lesson 14/1: Saving Classifiers with NLTK
# =============================================================================
#Based on lesson 11-13
#we trained our classifier in the previous lesson now we wanna save it to use it whenever we want
"""
import nltk
import random
from nltk.corpus import movie_reviews
import pickle

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

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Algo Accuracy Percentage:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

#created a file name naivebayes.pickle; wb is write in bites
save_classfiier = open ("Naivebayes.pickle", "wb")
pickle.dump(classifier, save_classfiier)
save_classfiier.close()
"""

# =============================================================================
# Lesson 14/2: Saving Classifiers with NLTK
# =============================================================================
#Based on lesson 11-13
#now that we got the classifer saved we call it 
"""
import nltk
import random
from nltk.corpus import movie_reviews
import pickle

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

#open the classifier; rb to read and it was saved in bytes
classifier_f = open ("naivebayes.pickle", "rb")
classifier = pickle.load (classifier_f)
classifier_f.close()

print("Naive Bayes Algo Accuracy Percentage:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

"""

# =============================================================================
# Lesson 15: Scikit-Learn Sklearn with NLTK
# =============================================================================

#based on leeson 11-14
#each of the below algorithm has thier own parameters; here they used in a simple way
"""
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB,BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

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
"""

# =============================================================================
# Lesson 16: Combining Algorithms with NLTK + Voting
# =============================================================================

#Based on Lesson 11-15
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








