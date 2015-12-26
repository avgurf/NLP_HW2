##Our import:
import nltk
from nltk.probability import ConditionalFreqDist
import pandas as pd
import math
from nltk import word_tokenize
import numpy
import numpy as np
import re
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from pandas import DataFrame
import sys

##Our defines:
q1Verbose = 1

##############################################       Question 1     ####################################################

# print("Question1\n" +
#       "---------\n\t")
#
# # Taken from import example-application-plot-out-of-core-classification.pt so we could call it's functions
# from plot_out_of_core_classification import *
#
#
# # From the code, accessing the reuters document data base.
# data_stream = stream_reuters_documents()
# # Experimenting with functions given in hint.
# df = pd.DataFrame(data_stream)
# print(df)
#
# #### How many documents in the dataset?
# print("Q1.1.2 - There are %d documents in our data set", df['title'].describe()['count'])
#
# # Create a list of all occurences of all topics and feed to FreqDist.
# freq_dist = nltk.FreqDist(sum(list(df['topics']), []))
# #### How many categories:
# category_set = set(sum(list(df['topics']), []))
# num_of_categories = len(category_set)
# print("The number of categories is: ", num_of_categories)
#
# #### How many documents per category (first 10):
# cat_numOfDocs = [(category, freq_dist[category]) for category in category_set]
# for pair in cat_numOfDocs[:10]:
#     print('Category: ', pair[0], 'has ', pair[1], 'Docs')
#
# ####  Provide mean and standard deviation, min and max.
# # Mean:
# # Sum of number of documents per each category.
# sum_docs_cat = sum(num_of_docs for (cat, num_of_docs) in cat_numOfDocs)
# print('The Average number of documents per categories is: ', sum_docs_cat / len(cat_numOfDocs))
# # Max:
# print('The categories with maximum documents is: "', freq_dist.max(), '"which has ', freq_dist[freq_dist.max()],
#       ' documents.')
#
# # Min:
# min_num_of_docs = sorted(cat_numOfDocs, key=lambda x: x[1])[0][1]
# cats_w_min_num_of_docs = [cat for (cat, num_of_docs) in cat_numOfDocs if num_of_docs == min_num_of_docs]
# display = 3  # Display only part of categories, not all.
# print('The category with minimum documents are:', cats_w_min_num_of_docs[:display], 'who have', min_num_of_docs,
#       'documents each. ')
#
# # Standard deviation:
# std_dev = math.sqrt(
#     sum((math.pow(num_of_docs - mean_exp, 2) for (_, num_of_docs) in cat_numOfDocs)) / len(cat_numOfDocs))
# print('The standard deviation in number of documents per category is:', std_dev)
#
# #### (1.1.3)  Explore how many characters and words are present in the documents of the dataset.
#
# #Takes a while to run, use with care :)
# word_set=set()
# for i in range(len(df['body'])):
#     word_set.update(word_tokenize(df['body'][i]))
#


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
        'POS=': postag,
        'ORT=': get_ort(word),
        'PREFIX1=': word.lower()[:1],
        'PREFIX2=': word.lower()[:2],
        'PREFIX3=': word.lower()[:3],
        'SUFFIX1=': word.lower()[-1:],
        'SUFFIX2=': word.lower()[-2:],
        'SUFFIX3=': word.lower()[-3:]}
    return features


def word2label(sent, i):
    return sent[i][2]


def generate_unk_feature():
    return {'WORD-FROM': 'UNK'}


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def senteces2vector(sents):
    v = DictVectorizer(sparse=False)
    features = [sent2features(sent) for sent in sents]
    features = sum(features, [])
    features.append(generate_unk_feature())
    return v.fit_transform(features)


def build_pipeline():
    pipeline = Pipeline([
        ('vectorize', DictVectorizer(sparse=False)),
        ('classify', LogisticRegression())
    ])
    return pipeline


train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))


def clear_nones(list):
    return [item for item in list if item is not None]


def build_data_frame(l, len_data, sent):
    rows = []
    index = []
    for i in range(len(sent)):
        rows.append({'features': word2features(sent, i), 'class': word2label(sent, i)})
        index.append(sent)
    progress(l, len_data)
    data_frame = DataFrame(rows, index=index)
    return data_frame, len(rows)


def progress(i, end_val, bar_length=50):
    '''
    Print a progress bar of the form: Percent: [#####      ]
    i is the current progress value expected in a range [0..end_val]
    bar_length is the width of the progress bar on the screen.
    '''
    percent = float(i) / end_val
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\rPercent: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
    sys.stdout.flush()


def build_data(data):
    df = DataFrame({'features': [], 'class': []})
    print("Starting To Build Data.")
    for i, sent in enumerate(data):
        data_frame, nrows = build_data_frame(i, len(data), sent)
        df = df.append(data_frame)
    return df


# This train function is based on the training
# of the spam classifier
def train(data_sents=None, data_frame=None, n_folds=6):
    if data_frame is None and data_sents is None:
        raise Exception('No data was provided to train!')
    elif data_frame is None:
        data_frame = build_data(data_sents)

    k_fold = KFold(n=len(data_frame), n_folds=n_folds)
    pipeline = build_pipeline()
    scores = []
    confusion = numpy.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])

    print("Training with %d folds" % n_folds)
    for i, (train_indices, test_indices) in enumerate(k_fold):
        x_train = data_frame.iloc[train_indices]['features'].values
        y_train = data_frame.iloc[train_indices]['class'].values.astype(str)
        x_test = data_frame.iloc[test_indices]['features'].values
        y_test = data_frame.iloc[test_indices]['class'].values.astype(str)

        print("Training for fold %d" % i)
        pipeline.fit(x_train, y_train)
        print("Testing for fold %d" % i)
        predictions = pipeline.predict(x_test)
        confusion += confusion_matrix(y_test, predictions)
        score = f1_score(y_test, predictions)
        scores.append(score)
        print("Score for %d: %2.2f" % (i, score))
        print("Confusion matrix for %d: " % i)
        print(confusion_matrix(y_test, predictions))
    print('Total classified:', len(data_frame))

    if len(scores)>0:
        print('Score:', sum(scores) / len(scores))
    print('Confusion matrix:')
    confusion(confusion)
    return pipeline, data_frame


model, data = train(train_sents[:200])
x_test = sum([sent2features(s) for s in test_sents], [])
y_test = sum([sent2labels(s) for s in test_sents], [])
predictions = model.predict(x_test)
score = f1_score(y_test, predictions)
##############################################        End of Q2     ####################################################
