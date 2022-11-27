# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Constants & Files
token_spam_prob_file = 'SpamData\\03_Testing\\prob-spam.txt'
token_ham_prob_file = 'SpamData\\03_Testing\\prob-ham.txt'
token_all_prob_file = 'SpamData\\03_Testing\\prob-all-tokens.txt'

test_feature_matrix = 'SpamData\\03_Testing\\test-features.txt'
test_target_file = 'SpamData\\03_Testing\\test-target.txt'

vocab_size = 2500

# Load the data
# Features
x_test = np.loadtxt(test_feature_matrix, delimiter=' ')

# Target
y_test = np.loadtxt(test_target_file, delimiter=' ')

# Token Probabilities
prob_token_spam = np.loadtxt(token_spam_prob_file, delimiter=' ')
prob_token_ham = np.loadtxt(token_ham_prob_file, delimiter=' ')
prob_all_tokens = np.loadtxt(token_all_prob_file, delimiter=' ')

# Joint Probability
# P(A AND B) = P(A) X P(B)

# Independence
# P(Spam | Tokens) = (P(Tokens | Spam) / P(Tokens)) X P(Spam)

# Set the prior
# P(Spam|X) = (P(X|Spam)*P(Spam)) / P(X)
prob_spam = 0.3116

# Joint Probability in log format
joint_log_spam = x_test.dot(np.log(prob_token_spam) - np.log(prob_all_tokens)) + np.log(prob_spam)
joint_log_ham = x_test.dot(np.log(prob_token_ham) - np.log(prob_all_tokens)) + np.log(1-prob_spam)

# Making Predictions
# Checking for the higher joint probability
# P(spam|x) > P(ham|x)
#         OR
# P(spam|x) < P(ham|x)

# Prediction vector
prediction = joint_log_spam > joint_log_ham
# The way we convert boolean to int
# prediction*1

# Simplify (Removing P(X) which is np.log(prob_all_tokens))
joint_log_spam = x_test.dot(np.log(prob_token_spam)) + np.log(prob_spam)
joint_log_ham = x_test.dot(np.log(prob_token_ham)) + np.log(1-prob_spam)

# Model Evaluation
# Metrics and Evaluation
# Accuracy Metric
# Correct predicted docs
correct_docs = (y_test == prediction).sum()
# Wrong predicted docs
numdocs_wrong = x_test.shape[0] - correct_docs
# Calculate the accuracy
accuracy = correct_docs/len(x_test)
# Docs classified incorrectly
fraction_wrong = numdocs_wrong/len(x_test)
print('Fraction classified incorrectly is {:.2%}'.format(fraction_wrong))
print('Accuracy of the model is {:.2%}'.format(1-fraction_wrong))


# False Positive and False Negative
# 1. False Positive: predicted a spam, but it's classified as nonspam
# The nonspam message ends up in spam folder
# 2. False Negative: predicted a nonspam but classified as a spam
# The spam message ends up in non spam folder
# 3. True Positive: predicted spam ,and it's spam
# 4. True Negative: predicted nonspam, and it's nonspam

# Calculate TP / FP / TN / FN
true_pos = (y_test == 1) & (prediction == 1)
# print(true_pos.sum())
false_pos = (y_test == 0) & (prediction == 1)
# print(false_pos.sum())
true_neg = (y_test == 0) & (prediction == 0)
# print(true_neg.sum())
false_neg = (y_test == 1) & (prediction == 0)
# print(false_neg.sum())

# The Recall Metric
# True Positives / (True Positives + False Negatives)
recall_score = true_pos.sum() / (true_pos.sum()+false_neg.sum())
print('Recall score {:.2%}'.format(recall_score))

# The Precision Metric
# True Positives / (True Positives + False Positives)
precision_score = true_pos.sum() / (true_pos.sum() + false_pos.sum())
print('Precision score {:.3}'.format(precision_score))

# F-Score OR F1 Score Metric
# 2 x ((Precision x Recall)/(Precision + Recall))
f1_score = 2 * ((precision_score*recall_score)/(precision_score+recall_score))
print('F1 score {:.2}'.format(f1_score))

# Visualising the Results
# Chart Styling Info
yaxis_label = 'P(X|Spam)'
xaxis_label = 'P(X|Nonspam)'
linedata = np.linspace(start=-14000, stop=1, num=1000)

plt.figure(figsize=(11, 7))
plt.xlabel(xaxis_label, fontsize=14)
plt.ylabel(yaxis_label, fontsize=14)
# Set Scale
plt.xlim([-14000, 1])
plt.ylim([-14000, 1])
plt.scatter(joint_log_ham, joint_log_spam, color='navy')
plt.show()

# The Decision Boundary
plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)
plt.xlabel(xaxis_label, fontsize=14)
plt.ylabel(yaxis_label, fontsize=14)
# Set Scale
plt.xlim([-14000, 1])
plt.ylim([-14000, 1])
plt.scatter(joint_log_ham, joint_log_spam, color='navy', alpha=0.5, s=25)
plt.plot(linedata, linedata, color='orange')

# Chart 2
plt.subplot(1, 2, 2)
plt.xlabel(xaxis_label, fontsize=14)
plt.ylabel(yaxis_label, fontsize=14)
# Set Scale
plt.xlim([-2000, 1])
plt.ylim([-2000, 1])
plt.scatter(joint_log_ham, joint_log_spam, color='navy', alpha=0.5, s=3)
plt.plot(linedata, linedata, color='orange')
plt.show()

# Chart Styling
sns.set_style('whitegrid')
labels = 'Actual Category'

summary_df = pd.DataFrame({yaxis_label: joint_log_spam, xaxis_label: joint_log_ham, labels: y_test})

sns.lmplot(x=xaxis_label, y=yaxis_label, data=summary_df, aspect=6.5, fit_reg=False,
           scatter_kws={'alpha': 0.7, 's': 25},
           hue=labels, markers=['o', 'x'], palette='hls', legend=False)

plt.xlim([-500, 1])
plt.ylim([-500, 1])
# Decision Boundary
plt.plot(linedata, linedata, color='red')
plt.legend(('Nonspam', 'Spam', 'Decision Boundary'),
           loc='lower right', fontsize=14)
plt.show()
