# Import Libraries
from os import walk
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from PIL import Image
from sklearn.model_selection import train_test_split

# Constants
# File paths
spam_1_path = 'SpamData\\01_Processing\\spam_assassin_corpus\\spam_1'
spam_2_path = 'SpamData\\01_Processing\\spam_assassin_corpus\\spam_2'
easy_non_spam_1_path = 'SpamData\\01_Processing\\spam_assassin_corpus\\easy_ham_1'
easy_non_spam_2_path = 'SpamData\\01_Processing\\spam_assassin_corpus\\easy_ham_2'

data_json_file = 'SpamData\\01_Processing\\email-text-data.json'
word_id_file = 'SpamData\\01_Processing\\word-by-id.csv'
training_data_file = 'SpamData\\02_Training\\train-data.txt'
testing_data_file = 'SpamData\\02_Training\\test-data.txt'

whale_file = 'SpamData\\01_Processing\\wordcloud_resources\\whale-icon.png'
thumbs_up_file = 'SpamData\\01_Processing\\wordcloud_resources\\thumbs-up.png'
thumbs_down_file = 'SpamData\\01_Processing\\wordcloud_resources\\thumbs-down.png'

# Categories and Vocabulary constants
spam_cat = 1
ham_cat = 0
vocab_size = 2500


# Email body extraction


def email_body_generator(path):
    # walk: generate file names in a directory by
    # walking the tree from the top to the bottom,
    # and it yields a tuple consisting of directory
    # path (root), directory names and file names
    for root, dirnames, filenames in walk(path):
        for file_name in filenames:
            filepath = join(root, file_name)
            stream = open(filepath, encoding='latin-1')
            is_body = False
            lines = []
            for line in stream:
                if is_body:
                    lines.append(line)
                elif line == '\n':
                    is_body = True
            stream.close()
            email_body = '\n'.join(lines)  # join: makes the list readable
            yield file_name, email_body


# Create DataFrame from directory


def df_from_directory(path, classification):  # df: data frame
    rows = []
    row_names = []
    for file_name, email_body in email_body_generator(path):
        # we create directory
        rows.append({'Message': email_body, 'Category': classification})
        row_names.append(file_name)
    return pd.DataFrame(rows, index=row_names)


# Create Data from Directory (Spam and Ham emails)


spam_emails = df_from_directory(spam_1_path, 1)
spam_emails = spam_emails.append(df_from_directory(spam_2_path, 1))
ham_emails = df_from_directory(easy_non_spam_1_path, ham_cat)
ham_emails = ham_emails.append(df_from_directory(easy_non_spam_2_path, ham_cat))

# Complete data (Concatenation of Spam and Ham emails)
data = pd.concat([spam_emails, ham_emails])

# STEP BY STEP EXPLANATION FOR DATA CLEANING
# 1. Checking for missing values
# check if any message bodies are null
# print(data.Message)
# isnull: return True or False values
# print(data['Message'].isnull())

# values: gives only the value True or false
# print(data['Message'].isnull().values)

# any: return False when there are no null values
# and True when there are null values
# print(data['Message'].isnull().values.any())

# 2. Check if there are empty emails (string length zero)
# print((data.Message.str.len() == 0).any())
# if we want to know how many emails ary empty
# we will sum them up using sum.
# print((data.Message.str.len() == 0).sum())

# 3. Check the number of entries with None values
# print((data.Message.isnull() == True).sum())
# print(data.Message.isnull().sum())

# 4. Locate empty emails
# 4-1. Check the type of the emails
# print(type(data.Message.str.len() == 0))
# 4-2. Check the index where there are empty emails
# print(data[data.Message.str.len() == 0].index)
# 4-3. Get the location of the emails
# print(data.index.get_loc('.DS_Store'))
# 4-4. Remove System file entries from dataframe
# data = data.drop(['cmds'])

# Instead, we can use inplace and will be updated
data.drop(['cmds'], inplace=True)

# Add Document IDs to track emails in dataset
document_ids = range(0, len(data.index))

# Make new column for IDs
data['Doc_Id'] = document_ids

# Shift the columns
data['File_Name'] = data.index
data.set_index('Doc_Id', inplace=True)

# Save the Data to file using JSON
data.to_json(data_json_file)

###################################
# Data Visualization
# Number of spam messages visualised (Pie charts)
# print(data.Category.value_counts())
amount_of_spam = data.Category.value_counts()[1]
amount_of_ham = data.Category.value_counts()[0]

# Chart 1 pie chart
# Create list of names
category_names = ['Spam', 'Legit Mail']
sizes = [amount_of_spam, amount_of_ham]
custom_colours = ['#ff7675', '#74b9ff']

# show the chart
plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names,
        textprops={'fontsize': 6},
        # to rotate the circle
        startangle=90,
        # for the percentage
        autopct='%1.1f%%',
        colors=custom_colours,
        # makes gap between the circle pieces
        explode=[0, 0.1])
plt.show()

# Chart 2 Donut chart
# Create list of names
category_names = ['Spam', 'Legit Mail']
sizes = [amount_of_spam, amount_of_ham]
custom_colours = ['#ff7675', '#74b9ff']

# show the chart
plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names,
        textprops={'fontsize': 6},
        # to rotate the circle
        startangle=90,
        # for the percentage
        autopct='%1.1f%%',
        colors=custom_colours,
        # make percentage move
        pctdistance=0.8)

# Draw a circle
Centre_circle = plt.Circle((0, 0),
                           # make circle
                           radius=0.6,
                           # color circle
                           fc='white')
# gca: get current axes
plt.gca().add_artist(Centre_circle)
plt.show()

# Chart 3 Donut chart with more categories
category_names = ['Spam', 'Legit Mail', 'Updates', 'Promotions']
sizes = [25, 43, 19, 22]
custom_colours = ['#ff7675', '#74b9ff', '#55efc4', '#ffeaa7']
offset = [0.05, 0.05, 0.05, 0.05]

# show the chart
plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names,
        textprops={'fontsize': 6},
        # to rotate the circle
        startangle=90,
        # for the percentage
        autopct='%1.1f%%',
        colors=custom_colours,
        # make percentage move
        pctdistance=0.8,
        explode=offset)

# Draw a circle
Centre_circle = plt.Circle((0, 0),
                           # make circle
                           radius=0.6,
                           # color circle
                           fc='white')
# gca: get current axes
plt.gca().add_artist(Centre_circle)
plt.show()

# END OF DATA VISUALIZATION
###################################

# NLP (Natural Language Processing)
# Pre-Processing our data:
# 1. Converting to lower case
# 2. Tokenizing
# 3. Removing stop words
# 4. Stripping out HTML tags
# 5. Word Stemming
# 6. Removing punctuation


# Function for email processing


def clean_message(message):
    filtered_words = []
    # Save stop words of english language into a variable
    stop_words = set(stopwords.words('english'))
    # Save the stemmer in a variable
    stemmer = SnowballStemmer('english')
    # Remove HTML from the message and then extract the text only
    soup = BeautifulSoup(message, 'html.parser')
    message = soup.get_text()
    # Tokenizing and Lowering the case
    words = word_tokenize(message.lower())
    for word in words:
        # Check the word if it's stop word
        # isalpha(): checks if the word contains of characters (alphabet)
        if word not in stop_words and word.isalpha():
            # Stemming
            stemmed_word = stemmer.stem(word)
            filtered_words.append(stemmed_word)
    return filtered_words

# Apply cleaning message to all the data Messages


nested_list = data.Message.apply(clean_message)

# Using logic to slice dataframe
doc_ids_spam = data[data.Category == 1].index
doc_ids_ham = data[data.Category == 0].index

# Sub-setting a series with an index
nested_list_ham = nested_list.loc[doc_ids_ham]
nested_list_spam = nested_list.loc[doc_ids_spam]

# Spam words
words_spam = [item for subset in nested_list_spam for item in subset]
# value_counts() removes the repetition
spam_words = pd.Series(words_spam).value_counts()

# Ham words
words_ham = [item for subset in nested_list_ham for item in subset]
ham_words = pd.Series(words_ham).value_counts()

###################################
# Visualizations with word cloud
# Creating chart for our dataset
# Word cloud of Ham and spam messages
word_list_ham = [''.join(word) for word in words_ham]
ham_as_string = ' '.join(word_list_ham)

ham_icon = Image.open(thumbs_up_file)
ham_image_mask = Image.new(mode='RGB', size=ham_icon.size, color=(255, 255, 255))
ham_image_mask.paste(ham_icon, box=ham_icon)
# Convert the image obj into array
ham_rgb_array = np.array(ham_image_mask)
ham_word_cloud = WordCloud(mask=ham_rgb_array, background_color='white', max_words=400, colormap='ocean')
ham_word_cloud.generate(ham_as_string)
plt.figure(figsize=[8, 8])
plt.imshow(ham_word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

word_list_spam = [''.join(word) for word in words_spam]
spam_as_string = ' '.join(word_list_spam)

spam_icon = Image.open(thumbs_down_file)
spam_image_mask = Image.new(mode='RGB', size=spam_icon.size, color=(255, 255, 255))
spam_image_mask.paste(spam_icon, box=spam_icon)
# Convert the image obj into array
spam_rgb_array = np.array(spam_image_mask)
spam_word_cloud = WordCloud(mask=spam_rgb_array, background_color='white', max_words=400, colormap='bone')
spam_word_cloud.generate(spam_as_string)
plt.figure(figsize=[8, 8])
plt.imshow(spam_word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# END OF DATA VISUALIZATION
###################################

# Generate Vocabulary and Dictionary
stemmed_nested_list = data.Message.apply(clean_message)
flat_stemmed_list = [item for sublist in stemmed_nested_list for item in sublist]
unique_words = pd.Series(flat_stemmed_list).value_counts()
frequent_words = unique_words[0:vocab_size]

# Create Vocabulary dataframe with word_id
word_ids = list(range(0, vocab_size))
vocab = pd.DataFrame({'Vocab_Words': frequent_words.index.values}, index=word_ids)
vocab.index.name = 'Word_ID'

# Save vocabulary as CSV file
vocab.to_csv(word_id_file, index_label=vocab.index.name, header=vocab.Vocab_Words.name)

clean_email_lengths = [len(sublist) for sublist in stemmed_nested_list]
# stemmed_nested_list[np.argmax(clean_email_lengths)]

# Features as Sparse Matrix
# Creating a dataframe with one word per column
# We use stemmed_nested_list
# Convert to list
word_columns_df = pd.DataFrame.from_records(stemmed_nested_list.tolist())

# Splitting the data into training and testing datasets
# Using sklearn package
x_train, x_test, y_train, y_test = train_test_split(word_columns_df, data.Category, test_size=0.3, random_state=42)
x_train.index.name = x_test.index.name = "doc_id"

# Creating sparse matrix for the training data
# Word ids
word_index = pd.Index(vocab.Vocab_Words)


def make_sparse_matrix(df, indexed_words, labels):
    """
    :param df: a dataframe with words in the column's id as an index (x_train or x_test)
    :param indexed_words: index of words ordered by word_id
    :param labels: category as a series (y_train or y_text)
    :return: sparse matrix as dataframe
    """
    nr_rows = df.shape[0]
    nr_cols = df.shape[1]
    word_set = set(indexed_words)
    dict_list = []

    for i in range(nr_rows):
        for j in range(nr_cols):
            word = df.iat[i, j]
            if word in word_set:
                doc_id = df.index[i]
                word_id = indexed_words.get_loc(word)
                category = labels.at[doc_id]
                item = {'LABEL': category, 'DOC_ID': doc_id, 'OCCURRENCE': 1, 'WORD_ID': word_id}
                dict_list.append(item)

    return pd.DataFrame(dict_list)

# Data frame for train data


sparse_train_df = make_sparse_matrix(x_train, word_index, y_train)

# Combine occurrences with the pandas group by() method
train_grouped = sparse_train_df.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum()
train_grouped = train_grouped.reset_index()

# Saving training data as .txt file
np.savetxt(training_data_file, train_grouped, fmt='%d')

# Data frame for test data
sparse_test_df = make_sparse_matrix(x_test, word_index, y_test)

# Combine occurrences with the pandas group by() method
test_grouped = sparse_test_df.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum()
test_grouped = test_grouped.reset_index()

# Saving testing data as .txt file
np.savetxt(testing_data_file, test_grouped, fmt='%d')
