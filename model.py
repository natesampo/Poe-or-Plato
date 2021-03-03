import re
import collections
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
import numpy as np
import math
import random

PLATO = 0
POE = 1

def clean_text(s):
	s = s.lower()
	s = re.sub('\n',' ',s)
	s = re.sub('<br />',' ',s)
	s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
	s = re.sub(' +',' ',s)
	return s


def generate_ngrams(s, n):
	tokens = [token for token in s.split(" ") if token != ""]
	sequences = [tokens[i:] for i in range(n)]
	ngrams = zip(*sequences)
	return [" ".join(ngram) for ngram in ngrams]


train_data = []
train_labels = []
test_data = []
test_labels = []
for filename in os.listdir(os.getcwd() + '/train/plato'):
	if filename.endswith('.txt'):
		with open(os.getcwd() + '/train/plato/' + filename, 'r') as file:
			train_data.append(clean_text(file.read()))
			train_labels.append(PLATO)

for filename in os.listdir(os.getcwd() + '/train/poe'):
	if filename.endswith('.txt'):
		with open(os.getcwd() + '/train/poe/' + filename, 'r') as file:
			train_data.append(clean_text(file.read()))
			train_labels.append(POE)

for filename in os.listdir(os.getcwd() + '/test/plato'):
	if filename.endswith('.txt'):
		with open(os.getcwd() + '/test/plato/' + filename, 'r') as file:
			test_data.append(clean_text(file.read()))
			test_labels.append(PLATO)

for filename in os.listdir(os.getcwd() + '/test/poe'):
	if filename.endswith('.txt'):
		with open(os.getcwd() + '/test/poe/' + filename, 'r') as file:
			test_data.append(clean_text(file.read()))
			test_labels.append(POE)

print("Gathered Data")

ngramvectorizer = CountVectorizer(ngram_range=(7, 7))
ngramvectorizer.fit(train_data)
train_ngram = ngramvectorizer.transform(train_data)
test_ngram = ngramvectorizer.transform(test_data)

print("Calculated N-Grams")

model = MultinomialNB(alpha=1)
model.fit(train_ngram, train_labels)

print("Trained Model")

predicted_train = model.predict(train_ngram)
print(train_labels)
print(predicted_train)
print("Training accuracy: ", np.mean(predicted_train == train_labels))
predicted_test = model.predict(test_ngram)
print(test_labels)
print(predicted_test)
print("Testing accuracy: ", np.mean(predicted_test == test_labels))


bigramPlato = [generate_ngrams(book, 2) for book in train_data if train_labels[train_data.index(book)] == PLATO]
bigramPoe = [generate_ngrams(book, 2) for book in train_data if train_labels[train_data.index(book)] == POE]

bigramLookupPlato = {}
bigramLookupPoe = {}

for i in range(len(bigramPlato)-1):
	for j in range(len(bigramPlato[i])-1):
		w1 = bigramPlato[i][j].split(" ")[0]
		w2 = bigramPlato[i][j].split(" ")[1]

		if w1 not in bigramLookupPlato.keys():
			bigramLookupPlato[w1] = {w2:1}
		elif w2 not in bigramLookupPlato[w1].keys():
			bigramLookupPlato[w1][w2] = 1
		else:
			bigramLookupPlato[w1][w2] = bigramLookupPlato[w1][w2] + 1

for i in range(len(bigramPoe)-1):
	for j in range(len(bigramPoe[i])-1):
		w1 = bigramPoe[i][j].split(" ")[0]
		w2 = bigramPoe[i][j].split(" ")[1]

		if w1 not in bigramLookupPoe.keys():
			bigramLookupPoe[w1] = {w2:1}
		elif w2 not in bigramLookupPoe[w1].keys():
			bigramLookupPoe[w1][w2] = 1
		else:
			bigramLookupPoe[w1][w2] = bigramLookupPoe[w1][w2] + 1

curr_sequence = "he" # Starting word
wordsPlato = curr_sequence
for i in range(50):
	if curr_sequence not in bigramLookupPlato.keys():
		wordsPlato += '. '
		curr_sequence = 'the'
		wordsPlato += curr_sequence
	else: 
		possible_words = list(bigramLookupPlato[curr_sequence].keys())
		next_word = possible_words[random.randrange(len(possible_words))] #Randomly choose a word
		wordsPlato += ' ' + next_word
		curr_sequence = next_word

curr_sequence = "once" # Starting word
wordsPoe = curr_sequence
for i in range(50):
	if curr_sequence not in bigramLookupPoe.keys():
		wordsPoe += '. '
		curr_sequence = 'the'
		wordsPoe += curr_sequence
	else: 
		possible_words = list(bigramLookupPoe[curr_sequence].keys())
		next_word = possible_words[random.randrange(len(possible_words))] #Randomly choose a word
		wordsPoe += ' ' + next_word
		curr_sequence = next_word

print("Plato: " + wordsPlato)
print("Poe: " + wordsPoe)