##Our import:
import nltk
from nltk.probability import ConditionalFreqDist
import pandas as pd
from pandas import DataFrame
import math
from nltk import word_tokenize
import numpy as np
from random import randint
import csv
import numpy
import re

import sys

from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_extraction import DictVectorizer

import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve


##Our defines:
q1Verbose=1
q113_verbose=0
q2Verbose=1

##############################################       Question 1     ####################################################

print("Question1\n" +
      "---------\n\t")

# Taken from import example-application-plot-out-of-core-classification.pt so we could call it's functions
from plot_out_of_core_classification import *

# From the code, accessing the reuters document data base.
data_stream = stream_reuters_documents()
# Experimenting with functions given in hint.
df = pd.DataFrame(data_stream)
print(df)

#### How many documents in the dataset?
print("Q1.1.2 - There are %d documents in our data set", df['title'].describe()['count'])

# Create a list of all occurences of all topics and feed to FreqDist.
freq_dist = nltk.FreqDist(sum(list(df['topics']), []))
#### How many categories:
category_set = set(sum(list(df['topics']), []))
num_of_categories = len(category_set)
print("The number of categories is: ", num_of_categories)

#### How many documents per category (first 10):
cat_numOfDocs = [(category, freq_dist[category]) for category in category_set]
for pair in cat_numOfDocs[:10]:
    print('Category: ', pair[0], 'has ', pair[1], 'Docs')

####  Provide mean and standard deviation, min and max.
# Mean:
# Sum of number of documents per each category.
sum_docs_cat = sum(num_of_docs for (cat, num_of_docs) in cat_numOfDocs)
print('The Average number of documents per categories is: ', sum_docs_cat / len(cat_numOfDocs))
# Max:
print('The categories with maximum documents is: "', freq_dist.max(), '"which has ', freq_dist[freq_dist.max()],
      ' documents.')

# Min:
min_num_of_docs = sorted(cat_numOfDocs, key=lambda x: x[1])[0][1]
cats_w_min_num_of_docs = [cat for (cat, num_of_docs) in cat_numOfDocs if num_of_docs == min_num_of_docs]
display = 3  # Display only part of categories, not all.
print('The category with minimum documents are:', cats_w_min_num_of_docs[:display], 'who have', min_num_of_docs,
      'documents each. ')

# Standard deviation:
std_dev = math.sqrt(
    sum((math.pow(num_of_docs - mean_exp, 2) for (_, num_of_docs) in cat_numOfDocs)) / len(cat_numOfDocs))
print('The standard deviation in number of documents per category is:', std_dev)

#### (1.1.3)  Explore how many characters and words are present in the documents of the dataset.

# Create sets of words and characters.
# Takes a while to run, use with care :)
if q113_verbose:
    word_set = set()
    word_list = []
    for i in range(len(df['body'])):
        word_set.update(word_tokenize(df['body'][i]))
        word_list += word_tokenize(df['body'][i])

    char_set = set()
    char_list = []
    for word in word_set:
        for letter in word:
            char_set.update(letter)
            char_list += letter
    print('There are %d different words in all documents. ' % len(word_set))
    print('There are %d word tokens in all documents. ' % len(word_list))
    print('There are %d different characters in all documents. ' % len(char_set))
    print('There are %d characters in all documents. ' % len(char_list))

# We will now construct a dictionary, That maps from article index to {num_of_words: , num_of_chars: }
article_2words_chars = {}
for i in range(len(df['body'])):
    article_2words_chars[i] = (len(word_tokenize(df['body'][i])), len(df['body'][i]))


def explore_doc(i):
    print(
    'Document with index %d has %d words and %d letters' % (2, article_2words_chars[x][0], article_2words_chars[x][1]))


#### (1.1.4) Explain informally what are the classifiers that support the "partial-fit" method discussed in the code.
# Informally, the classifiers that support "partial-fit",
# are classifiers who do not need to "hold" all the information
# they are given, at every given moment. If we attempt
# a slightly more formal explanation,
#  We can say that the state of the classifier is changed
# as it learns from more inputs, yet this input is not a state variable.

#### (1.1.5) Explain what is the hashing vectorizer used in this tutorial
# Why is it important to use this vectorizer to achieve "streaming classification"?
# As We have seen, We are dealing with a large amount of data.
# In order to make our data easier to process, We turn it into a
# sparse matrix that improves our memory usage by changing words into corresponding integers.

#### (1.2) Spam Dataset

# Code taken from the spam notebook
import os
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score

NEWLINE = '\n'

HAM = 'ham'
SPAM = 'spam'

SOURCES = [
    ('data/spam', SPAM),
    ('data/easy_ham', HAM),
    ('data/hard_ham', HAM),
    ('data/beck-s', HAM),
    ('data/farmer-d', HAM),
    ('data/kaminski-v', HAM),
    ('data/kitchen-l', HAM),
    ('data/lokay-m', HAM),
    ('data/williams-w3', HAM),
    ('data/BG', SPAM),
    ('data/GP', SPAM),
    ('data/SH', SPAM)
]

SKIP_FILES = {'cmds'}


def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    past_header, lines = False, []
                    f = open(file_path, encoding="latin-1")
                    for line in f:
                        if past_header:
                            lines.append(line)
                        elif line == NEWLINE:
                            past_header = True
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path, content


def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame


data = DataFrame({'text': [], 'class': []})
for path, classification in SOURCES:
    data = data.append(build_data_frame(path, classification))

data = data.reindex(numpy.random.permutation(data.index))

pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])

k_fold = KFold(n=len(data), n_folds=6)
scores = []
confusion = numpy.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['class'].values.astype(str)

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['class'].values.astype(str)

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label=SPAM)
    scores.append(score)

print('Total emails classified:', len(data))
print('Score:', sum(scores) / len(scores))
print('Confusion matrix:')
print(confusion)

# Total emails classified: 32389
# Score: 0.970473398143
# Confusion matrix:
# [[21829    60]
#  [  546  9954]]

#### (1.2.1) The vectorizer used in Zac Stewart's code is a CountVectorizer with unigrams


#  and bigrams. Report the number of unigrams and bigrams used in this model.
# Retreive the count vectorizer used in the model.
p = pipeline.get_params()
CountV = p['count_vectorizer']
# Access features:
uni_bi_grams = CountV.get_feature_names()
print("There are %d unigrams and bigrams, used in this model. " % len(uni_bi_grams))


# There are 1984848 unigrams and bigrams, used in this model.

#### (1.2.2) What are the 50 most frequent unigrams and bigrams in the dataset?
def most_freq_feat(classifier, count_vector, n=50):
    index = 0
    features_c1_c2_count = []

    for feat, c1, c2 in zip(count_vector.get_feature_names(), classifier.feature_count_[0],
                            classifier.feature_count_[1]):
        features_c1_c2_count.append((feat, c1 + c2))
        index += 1

    for i in sorted(features_c1_c2_count, key=lambda x: x[1], reverse=True)[:n]:
        print(i)


most_freq_feat(p['classifier'], p['count_vectorizer'], n=3)


# ('the', 274312.0)
# ('to', 190011.0)
# ('and', 140757.0)

# What are the 50 most frequent unigrams and bigrams per class (ham and spam)?

# Create a list of feature name and amount of occurrences in each class.
# Sort according to different class counter to get occurrences per class.
def most_occurring_feat_per_class(classifier, count_vector, n=50):
    index = 0
    features_c1_c2_count = []

    for feat, c1, c2 in zip(count_vector.get_feature_names(), classifier.feature_count_[0],
                            classifier.feature_count_[1]):
        features_c1_c2_count.append((feat, c1, c2))
        index += 1

    print("%d most occurring features in class spam: " % n)
    for i in sorted(features_c1_c2_count, key=lambda x: x[1], reverse=True)[:n]:
        print(i)

    print("%d most occurring features in class ham: " % n)
    for i in sorted(features_c1_c2_count, key=lambda x: x[2], reverse=True)[:n]:
        print(i)


most_occurring_feat_per_class(p['classifier'], p['count_vectorizer'], n=1)


# 1 most occurring features in class spam:
# ('the', 205629.0, 68683.0)
# 1 most occurring features in class ham:
# ('font', 11189.0, 89667.0)

#### (1.2.4) List the 20 most useful features in the Naive Bayes classifier to
#            distinguish between spam and ham (20 features for each class).

# Since each features coefficient links it to it's class, and smaller coefficients classify spam and larger ham,
# we sort according to coefficient, once normaly and once reversed, to get most informative features.
def most_informative_feature_for_binary_classification(vectorizer, classifier, n=20):
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    counter = 0
    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)
        if counter == 20:
            break
    print()

    counter = 0
    for coef, feat in reversed(topn_class2):
        print(class_labels[1], coef, feat)
        counter += 1
        if counter == 20:
            break


most_informative_feature_for_binary_classification(p['count_vectorizer'], p['classifier'], n=2)


# ham -16.0769551682 00 005
# ham -16.0769551682 00 00am
#
# spam -4.67308592853 font
# spam -4.80892052145 br

#### (1.2.5) There seems to be an imbalance in the length of spam and ham messages
#  (see the plot in the attached notebook).
# We want to add a feature based on the number of words in the message in the text representation.
# Should the length attribute be normalized before fitting the Naive Bayes classifier?
#  (See Sklearn pre-processing for examples.)
# Do you expect Logistic Regression to perform better with the new feature? Explain.).

def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'len': len(nltk.tokenize.word_tokenize(text)), 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame


data1 = DataFrame({'text': [], 'len': [], 'class': []})
for path, classification in SOURCES:
    data1 = data1.append(build_data_frame(path, classification))

data1 = data1.reindex(numpy.random.permutation(data.index))

pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])

data1 = DataFrame({'text': [], 'len': [], 'class': []})
for path, classification in SOURCES:
    data1 = data1.append(build_data_frame(path, classification))

data1 = data1.reindex(numpy.random.permutation(data.index))

from sklearn.pipeline import FeatureUnion

pipeline = Pipeline([
    ('features', FeatureUnion([
        ('body_stats', Pipeline([
            ('stats', TextStats()),  # returns a list of dicts
            ('vect', DictVectorizer()),  # list of dicts -> feature matrix
        ])),
        ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
    ])),
    ('classifier', MultinomialNB())
])

k_fold = KFold(n=len(data), n_folds=6)
scores = []
confusion = numpy.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['class'].values.astype(str)

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['class'].values.astype(str)

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label=SPAM)
    scores.append(score)

print('Total emails classified:', len(data))
print('Score:', sum(scores) / len(scores))
print('Confusion matrix:')
print(confusion)

# Total emails classified: 32389
# Score: 0.971156501249
# Confusion matrix:
# [[21818    71]
#  [  522  9978]]

from sklearn.linear_model import LogisticRegression


def build_pipeline2():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('body_stats', Pipeline([
                ('stats', TextStats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),
            ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ])),
        ('classifier', LogisticRegression())
    ])
    return pipeline


##############################################        End of Q1     ####################################################




##############################################       Question 2     ####################################################

print("Question2\n" +
      "---------\n\t")


def get_ort(word):
    if (re.match("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", word) != None):
        return "URL"
    if (re.match("^-?\+?[0-9]+(.[0-9]+)?$", word) != None):
        return "number"
    if (re.search(".*[0-9]+.*", word) != None):
        return "contains-digit"
    if (word.find("-") != -1):
        return "contains-hyphen"
    if word.isupper():
        return "all-capitals"
    if (re.match("^[A-Z].*", word) != None):
        return "capitalized"
    if (re.match("^[,;.-/!/?/*/+]+$", word) != None):
        return "punctuation"
    return "regular"


# Now we'll define a function to create a features dictionary out of the dataset
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'WORD-FROM': word.lower(),
        'POS': postag,
        'ORT': get_ort(word),
        'PREFIX1': word.lower()[:1],
        'PREFIX2': word.lower()[:2],
        'PREFIX3': word.lower()[:3],
        'SUFFIX1': word.lower()[-1:],
        'SUFFIX2': word.lower()[-2:],
        'SUFFIX3': word.lower()[-3:]}
    return features


def word2label(sent, i):
    return sent[i][2]


# Build our pipeline, using the DictVectorizer
# and LogisticRegrestion classifier.
def build_pipeline():
    """
    Builds our greedy NER Tagger pipeline based on
    a DictVectorizer and a LogisticRegresstion Classifier
    """
    pipeline = Pipeline([
        ('vectorize', DictVectorizer(sparse=True)),
        ('classify', LogisticRegression())
    ])
    return pipeline


# Split the data-set to test and train (using the testb)
train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))


def progress(i, end_val, bar_length=50):
    """
    Print a progress bar of the form: Percent: [#####      ]
    i is the current progress value expected in a range [0..end_val]
    bar_length is the width of the progress bar on the screen.
    """
    percent = float(i) / end_val
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\rPercent: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
    sys.stdout.flush()


def build_data(data, feature_extractor=(lambda sent, i: word2features(sent, i))):
    """
    Builds our NER tag-set DataFrame from a list of sentences, extracts features and
     tags to {Freature:X, 'calss":Y} structure
    :param data: list of sentences, each sentece is a list containing triples of
     (<word-form>,<pos>,<NER-tag>)
    :param feature_extractor: a mapping function from a word
    (given by a sentence(=list of triples) and and index) to a dictionary-like list
    with features
    :return: DataFame with Freature:X, 'calss":Y} when 'Features' is a list of dictionaries
    with extracted feature data
    """
    df = DataFrame({'features': [], 'class': []})
    print("Starting To Build Data.")
    for i, sent in enumerate(data):
        data_frame, nrows = build_data_frame_for_sentence(i, len(data), sent, feature_extractor)
        df = df.append(data_frame)
    print("Done.")
    return df


def build_data_frame_for_sentence(l, len_data, sent, feature_extractor):
    """
    Consturcts a DataFarame for each sentence (provided by a list of WORD,POS,NER triples
    :param l: current sentence index
    :param len_data: total number of sentences
    :param sent: the sentence (list of triples)
    :param feature_extractor: a mapping function to dictionary of features
    :return: a DataFrame with list of dictionarys representing features for each word
    in the sentence and a list of NER tags for each word in the sentece
    """
    rows = []
    index = []
    for i in range(len(sent)):
        rows.append({'features': feature_extractor(sent, i), 'class': word2label(sent, i)})
        index.append(sent)
    progress(l, len_data)
    data_frame = DataFrame(rows, index=index)
    return data_frame, len(rows)


# Trains our greedy NER model
def train(data_sents=None, data_frame=None, n_folds=6):
    if data_frame is None and data_sents is None:
        raise Exception('No data was provided to train!')
    elif data_frame is None:
        data_frame = build_data(data_sents)

    k_fold = KFold(n=len(data_frame), n_folds=n_folds)
    pipeline = build_pipeline()
    scores = []
    print("Training with %d folds" % n_folds)
    for i, (train_indices, test_indices) in enumerate(k_fold):
        x_train = data_frame.iloc[train_indices]['features'].values
        y_train = data_frame.iloc[train_indices]['class'].values.astype(str)
        if (q2Verbose):
            print("Training for fold %d" % i)
        pipeline.fit(x_train, y_train)
        x_test = data_frame.iloc[train_indices]['features'].values
        y_test = data_frame.iloc[train_indices]['class'].values.astype(str)
        scores.append(pipeline.score(x_test, y_test))
    print('Total classified:', len(data_frame))
    summary = np.mean(scores)
    print ("(avrage score:)\n", summary)
    summary = []
    return pipeline, data_frame, summary


# Now we can play a bit with the features
def word2features2(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    fetures = {
        'WORD-FROM': word.lower(),
        'POS': postag,
        'ORT': get_ort(word),
        'PREFIX1': word.lower()[:1],
        'PREFIX2': word.lower()[:2],
        'PREFIX3': word.lower()[:3],
        'SUFFIX1': word.lower()[-1:],
        'SUFFIX2': word.lower()[-2:],
        'SUFFIX3': word.lower()[-3:]}

    if i < (len(sent) - 1):
        fetures['NEXT_WORD_FORM'] = sent[i + 1][0]
        fetures['NEXT_POS'] = sent[i + 1][1]
    else:
        fetures['NEXT_WORD_FORM'] = '*'
        fetures['NEXT_POS'] = '*'

    if i > 0:
        fetures['PREV_WORD_FORM'] = sent[i - 1][0]
        fetures['PREV_POS'] = sent[i - 1][1]
    else:
        fetures['PREV_WORD_FORM'] = '*'
        fetures['PREV_POS'] = '*'

    return fetures


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [word2label(sent, i) for i in range(len(sent))]


data1 = build_data(train_sents[:20])
data2 = build_data(train_sents[:20], word2features2)

model1, data1, stats = train(None, data1)
model2, data2, stats = train(None, data2)

x_test = sum([sent2features(s) for s in test_sents], [])
y_test = sum([sent2labels(s) for s in test_sents], [])

print("testing model1..")
# Applies transforms to the data.
predictions1 = model1.predict(x_test)
score1 = f1_score(y_test, predictions1)

print("testing model2..")


def sent2features2(sent):
    return [word2features2(sent, i) for i in range(len(sent))]


x_test2 = sum([sent2features2(s) for s in test_sents], [])
predictions2 = model2.predict(x_test2)
score2 = f1_score(y_test, predictions2)

print("First model (without looking on previous and next word tags) scored %f " % score1)
print("After adding to the feature extraction better features we were managed to score  %f" % score2)

print ("Here is both confusions matrix of the first one:\n", confusion_matrix(y_test, predictions1))
print ("And Here is the second:\n", confusion_matrix(y_test, predictions2))

lbl_list = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O']

print("Model1: per-tag model1 statistics:\n")
prf1 = precision_recall_fscore_support(y_test, predictions1, labels=lbl_list)

print("-Per-tag model1 precision:\n")
for i, lbl in enumerate(lbl_list):
    print("Tag: \'{0}\' precision: {1}".format(lbl, prf1[0][i]))
print("\t---")

print("-Per-tag model1 recall:\n")
for i, lbl in enumerate(lbl_list):
    print('Tag: \'{0}\' recall: {1}'.format(lbl,prf1[1][i]))
print("\t---")

print("-Per-tag model1 f-score:\n")
for i, lbl in enumerate(lbl_list):
    print('Tag: \'{0}\' f-score: {1}'.format(lbl,prf1[2][i]))
print("\t---")

print("\tModel2: per-tag statistics:\n")
prf2 = precision_recall_fscore_support(y_test, predictions1, labels=lbl_list)

print("-Per-tag model2 precision:\n")
for i, lbl in enumerate(lbl_list):
    print('Tag: \'{0}\' precision: {1}'.format(lbl, prf2[0][i]))
print("\t---")

print("-Per-tag model2 recall:\n")
for i, lbl in enumerate(lbl_list):
    print('Tag: \'{0}\' recall: {1}'.format(lbl, prf2[1][i]))
print("\t---")

print("-Per-tag model2 f-score:\n")
for i, lbl in enumerate(lbl_list):
    print('Tag: \'{0}\' f-score: {1}'.format(lbl,prf2[2][i]))
print("\t---")

##############################################        End of Q2     ####################################################
