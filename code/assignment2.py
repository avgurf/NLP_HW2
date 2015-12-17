##Our import:
import nltk
from nltk.probability import ConditionalFreqDist
import pandas as pd
import math
from nltk import word_tokenize
import numpy as np

##Our defines:
q1Verbose = 1

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

#Takes a while to run, use with care :)
word_set=set()
for i in range(len(df['body'])):
    word_set.update(word_tokenize(df['body'][i]))



##############################################        End of Q1     ####################################################




##############################################       Question 2     ####################################################

print("Question2\n" +
      "---------\n\t")
##############################################        End of Q2     ####################################################
