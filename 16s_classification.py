# Import all the needed libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt

# Import the dataset (converted from fasta to CSV format)
Seq = pd.read_csv('16s_sequences_final.csv')
Seq['Class'].unique()

# Import label encoder (This is for the various microorganisms)
from sklearn import preprocessing
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'species'.
Seq['Class']= label_encoder.fit_transform(Seq['Class'])
Seq['Class'].unique()

# Inspect the first five lines
Seq.head()

# Visualize the distribution of the classes
Seq['Class'].value_counts().sort_index().plot.bar()
plt.title("Class distribution of 16s_Sequences")

# Convert a sequence of characters into k-mer words, default size = 6 (hexamers).
# The function Kmers_funct() will collect all possible overlapping k-mers of a
# specified length from any sequence string

def Kmers_funct(seq, size=6):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]
# convert the training data sequences into short overlapping k-mers of length 6.
# Let's do that for each species of data we have, using our Kmers_funct function.
Seq['words'] = Seq.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
Seq_dna = Seq.drop('sequence', axis=1)

# Visualize the first five entries
Seq_dna.head()

# We now need to convert the lists of k-mers for each gene into string sentences of words
# that can be used to create the Bag of Words model. We will make a target variable
# y to hold the class labels.

Seq_texts = list(Seq_dna['words'])
for item in range(len(Seq_texts)):
    Seq_texts[item] = ' '.join(Seq_texts[item])
#separate labels
y_Seq = Seq_dna.iloc[:, 1].values # y_Seq

# Visualize the labels
y_Seq

# Convert the k-mer words into uniform length numerical vectors that
# represent counts for every k-mer in the vocabulary

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(4,4)) #The n-gram size of 4 is previously determined by testing
X = cv.fit_transform(Seq_texts)

# Inspect the shape of the data
print(X.shape)
# There are 31 genes converted into uniform length
# feature vectors of 4-gram k-mer (length 6) counts.

# 80% and 20% of the data was used to train and test the classifiers, respectively
# test the model.
# Next, train/test split dataset and build simple multinomial naive Bayes classifier.
# Splitting the human dataset into the training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y_Seq,
                                                    test_size = 0.20,
                                                    random_state=42)

# Multinomial Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB
multi_class = MultinomialNB(alpha=0.1)
multi_class.fit(X_train, y_train)

# Now let’s make predictions for the Multinomial Naive Bayes Classifier
multi_pred = multi_class.predict(X_test)

# Performance metrics measurement: Multinomial Naive Bayes Classifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("Confusion matrix for predictions on human test DNA sequence\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(multi_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, multi_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

# Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(X_train.todense(), y_train)

# Now let’s make predictions for the Gaussian Naive Bayes Classifier
y_predGNB = GNB.predict(X_test.toarray())

# Performance metrics measurement: Gaussian Naive Bayes Classifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("Confusion matrix for predictions on human test DNA sequence\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_predGNB, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_predGNB)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

# Binomial Naive Bayes Classifier
from sklearn.naive_bayes import BernoulliNB
BNB = BernoulliNB()
BNB.fit(X_train, y_train)

# Now let’s make predictions for the Binomial Naive Bayes Classifier
y_predBNB = GNB.predict(X_test.toarray())

# Performance metrics measurement: Binomial Naive Bayes Classifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("Confusion matrix for predictions on human test DNA sequence\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_predBNB, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_predBNB)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

# Logistic regression Classifier
from sklearn.linear_model import LogisticRegression
logit_class = LogisticRegression()
logit_class.fit((X_train), y_train)

# Now let’s make predictions for the Logistic regression Classifier
logit_class = logit_class.predict(X_test.toarray())

# Performance metrics measurement: Logistic regression Classifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("Confusion matrix for predictions Logistic regression Classifier\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(logit_class, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, logit_class)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
