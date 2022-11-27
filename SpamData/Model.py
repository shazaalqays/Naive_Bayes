# Import Libraries
import pandas as pd
import numpy as np

# Constants & Files
training_data_file = 'SpamData\\02_Training\\train-data.txt'
testing_data_file = 'SpamData\\02_Training\\test-data.txt'

token_spam_prob_file = 'SpamData\\03_Testing\\prob-spam.txt'
token_ham_prob_file = 'SpamData\\03_Testing\\prob-ham.txt'
token_all_prob_file = 'SpamData\\03_Testing\\prob-all-tokens.txt'

test_feature_matrix = 'SpamData\\03_Testing\\test-features.txt'
test_target_file = 'SpamData\\03_Testing\\test-target.txt'

vocab_size = 2500

# Read and load features from txt files into numpy arrays
sparse_train_data = np.loadtxt(training_data_file, delimiter=' ', dtype=int)
sparse_test_data = np.loadtxt(testing_data_file, delimiter=' ', dtype=int)

# When visualizing the data it will be like following:
# First column is the doc_id (email)
# Second column is word_id (token)
# Third column is category (non spam = 0 / spam = 1)
# Fourth column is number of times the word occurred in the email
# print(sparse_train_data[:5])
# print('Nr of rows in training file', sparse_train_data.shape[0])
# print('Nr of rows in testing file', sparse_test_data.shape[0])
# print('Nr of unique emails in training file', np.unique(sparse_train_data[:,0]).size)
# print('Nr of unique emails in testing file', np.unique(sparse_test_data[:,0]).size)

# From sparse to full matrix
# How to create an empty data frame?
column_names = ['DOC_ID']+['CATEGORY'] + list(range(0, vocab_size))
# We define [:, 0] that we want the first column
index_names = np.unique(sparse_train_data[:, 0])
full_train_data = pd.DataFrame(index=index_names, columns=column_names)
# Inplace = make change to dataframe directly
full_train_data.fillna(value=0, inplace=True)

# Create a full matrix from a sparse matrix


def make_full_matrix(sparse_matrix, nr_words, doc_idx=0, word_idx=1, cat_idx=2, freq_idx=3):
    """
    Form a full matrix from a sparse matrix.
    Return a pandas dataframe.
    Keyword arguments:
    sparse_matrix: numpy array.
    nr_words: size of the vocabulary.
    doc_idx: position of the document id in the sparse matrix.
    word_idx: position of the word id in the sparse matrix.
    cat_idx: position of the category in the sparse matrix.
    freq_idx: position of the occurrence in the sparse matrix.
    """
    # Create the empty full matrix
    columns_names = ['DOC_ID'] + ['CATEGORY'] + list(range(0, nr_words))
    doc_id_names = np.unique(sparse_matrix[:, 0])
    full_matrix = pd.DataFrame(index=doc_id_names, columns=columns_names)
    full_matrix.fillna(value=0, inplace=True)

    # Populating the full matrix
    for i in range(sparse_matrix.shape[0]):
        # Getting the data from the sparse matrix
        doc_nr = sparse_matrix[i][doc_idx]
        word_id = sparse_matrix[i][word_idx]
        label = sparse_matrix[i][cat_idx]
        occurrence = sparse_matrix[i][freq_idx]

        # Selecting a particular cell in the dataframe
        # and fill the information
        full_matrix.at[doc_nr, 'DOC_ID'] = doc_nr
        full_matrix.at[doc_nr, 'CATEGORY'] = label
        full_matrix.at[doc_nr, word_id] = occurrence

    full_matrix.set_index('DOC_ID', inplace=True)

    return full_matrix

# Full matrix


full_train_data = make_full_matrix(sparse_train_data, vocab_size)

# Token Probabilities
# p(Spam|Viagra) = ((Occurrence in spam / Nr. of words in spam) x 0.55) / (Occurrence overall/ Total nr. of words)

# Training the Naive Bayes model
# Calculation the probability of spam
# Calculating the full data size
# print(full_train_data.CATEGORY.size)
# print(full_train_data.CATEGORY.sum())

# Probability of spam
prob_spam = full_train_data.CATEGORY.sum()/full_train_data.CATEGORY.size
# print('Probability of spam= ', prob_spam)

# The total number of words/ tokens
full_train_features = full_train_data.loc[:, full_train_data.columns != 'CATEGORY']
# axis = 1 will sum the rows
email_length = full_train_features.sum(axis=1)
total_wc = email_length.sum()

# Number of tokens in spam and ham emails
spam_length = email_length[full_train_data.CATEGORY == 1]
spam_wc = spam_length.sum()
ham_length = email_length[full_train_data.CATEGORY == 0]
non_spam_wc = ham_length.sum()

# Summing the tokens occurring in spam
train_spam_tokens = full_train_features.loc[full_train_data.CATEGORY == 1]
# axis = 0 will sum the columns
# Adding 1 avoid dividing by 0
summed_spam_tokens = train_spam_tokens.sum(axis=0) + 1

# Summing the tokens occurring in ham
train_ham_tokens = full_train_features.loc[full_train_data.CATEGORY == 0]
summed_ham_tokens = train_ham_tokens.sum(axis=0) + 1

# P(token | spam) probability that a token occurs given the email is spam
prob_tokens_spam = summed_spam_tokens / (spam_wc+vocab_size)
prob_tokens_spam.sum()

# P(token | ham) probability that a token occurs given the email is ham
prob_tokens_non_spam = summed_ham_tokens / (non_spam_wc+vocab_size)
prob_tokens_nonspam.sum()

# P(token) the probability that token occurs
prob_tokens_all = full_train_features.sum(axis=0) / total_wc
prob_tokens_all.sum()

# Save the Trained Model
np.savetxt(token_spam_prob_file, prob_tokens_spam)
np.savetxt(token_ham_prob_file, prob_tokens_non_spam)
np.savetxt(token_all_prob_file, prob_tokens_all)

# Prepare test data
# Same as did in training data
full_test_data = make_full_matrix(sparse_test_data, nr_words=vocab_size)

x_test = full_test_data.loc[:, full_test_data.columns != 'CATEGORY']
y_test = full_test_data.CATEGORY

np.savetxt(test_target_file, y_test)
np.savetxt(test_feature_matrix, x_test)
