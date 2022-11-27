# Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

DATA_JSON_FILE = 'SpamData/01_Processing/email-text-data.json'

# Import json file as a DataFrame
data = pd.read_json(DATA_JSON_FILE)
# data: 1.category - 2.message - 3.filename
# print(data.tail())
data.sort_index(inplace=True)
# print(data.tail())

# Create vocabulary
vectorizer = CountVectorizer(stop_words='english')
# document feature matrix
all_features = vectorizer.fit_transform(data.Message)
# print(all_features.shape)
# Vocabulary
# print(vectorizer.vocabulary_)

X_train, X_test, Y_train, Y_test = train_test_split(all_features, data.Category, test_size=0.3, random_state=88)

# Create the Model
classifier = MultinomialNB()
# Train the model
classifier.fit(X_train, Y_train)

nr_correct = (Y_test == classifier.predict(X_test)).sum()
# print(nr_correct)
nr_incorrect = Y_test.size - nr_correct
# print(nr_incorrect)

# Accuracy
fraction_wrong = nr_incorrect / (nr_correct+nr_incorrect)
# print('{:.2}'.format(1-fraction_wrong))
print('Accuracy: ', classifier.score(X_test, Y_test))
print('Recall: ', recall_score(Y_test, classifier.predict(X_test)))
print('Precision: ', precision_score(Y_test, classifier.predict(X_test)))
print('F1 score: ', f1_score(Y_test, classifier.predict(X_test)))

example = ['get viagra for free now!',
           'need a mortgage? reply to arrange a call with one',
           'could you please help me with the homework',
           'hello, how about a golf game',
           'Ski jumping is a winter sport in which competitors']

doc_term_matrix = vectorizer.transform(example)
print(classifier.predict(doc_term_matrix))
