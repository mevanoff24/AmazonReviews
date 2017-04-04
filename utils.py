import pandas as pd 
import numpy as np 
import re
import os 
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from tensorflow.contrib import learn


word_net_lematizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

def oneHot(y):
	N = len(y)
	K = len(set(y))
	ind = np.zeros((N, K))
	for i in range(N):
		ind[i, y[i]] = 1
	return ind

def clean_str(string):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()


def combine_scores(score):
	if score >= 4:
		return 2
	if score == 3:
		return 1
	else:
		return 0


def get_data(sample=False):
	t0 = datetime.now()
	dat = pd.read_csv('data/amazon-fine-foods/Reviews.csv', usecols=['Score', 'Text'])
		
	dat['Score'] = dat.Score.map(lambda x: combine_scores(x))
	dat['Text'] = dat.Text.map(lambda x: clean_str(x))

	positive = dat[dat['Score'] == 2][['Score', 'Text']]
	neutral = dat[dat['Score'] == 1][['Score', 'Text']]
	negative = dat[dat['Score'] == 0][['Score', 'Text']]
	del dat
	if sample:
		samples = sample
	else:
		samples = min(len(positive), len(neutral), len(negative))

	positive = positive.reindex(np.random.permutation(positive.index))[:samples]
	neutral = neutral.reindex(np.random.permutation(neutral.index))[:samples]
	negative = negative.reindex(np.random.permutation(negative.index))[:samples]
	assert(len(positive) == len(neutral) == len(negative))
	
	df = pd.concat([positive, neutral, negative])

	X = df.Text.values
	y = df.Score.values
	print 'Data Read Time', datetime.now() - t0
	return X, y

def get_data_straight(sample=True):
	if sample:
		dat = pd.read_csv('data/amazon-fine-foods/Reviews.csv', usecols=['Score', 'Text'], nrows=3000)
	else:
		dat = pd.read_csv('data/amazon-fine-foods/Reviews.csv', usecols=['Score', 'Text'])
	
	dat['Score'] = dat.Score.map(lambda x: combine_scores(x))
	dat['Text'] = dat.Text.map(lambda x: clean_str(x))
	X = dat.Text.values
	y = dat.Score.values
	return X, y



def my_tokenizer(s):
	tokens = word_tokenize(s)
	tokens = [word_net_lematizer.lemmatize(token) for token in tokens]
	tokens = [token for token in tokens if token not in STOPWORDS]
	return tokens



def accuracy(y_true, y_pred):
	return np.mean(y_true == y_pred)


def bag_of_words_data(sample=False):

	def _to_vector(tokens):
		X = np.zeros(len(word2idx))
		for token in tokens:
			token_index = word2idx[token]
			X[i] = 1
		return X

	# reviews, y = get_data()
	reviews, y = get_data_straight()
	print 'All Positve Benchmark', np.mean(y==2)

	word2idx = {}
	current_index = 0
	all_tokens = []

	for review in reviews:
		if review:
			tokens = my_tokenizer(review)
			all_tokens.append(tokens)
			for token in tokens:
				if token not in word2idx:
					word2idx[token] = current_index
					current_index += 1

	N = len(all_tokens)
	D = len(word2idx)
	X = np.zeros((N, D))
	i = 0
	for tokens in all_tokens:
		X[i, :] = _to_vector(tokens)
		i += 1

	print 'Data Loaded'
	return X, y



def get_sequence_data():
	# reviews, y = get_data()
	reviews, y = get_data_straight()
	# print 'All Positve Benchmark', np.mean(y==2)

	word2idx = {}
	all_tokens = []
	current_index = 0
	for review in reviews:
		if review:
			# tokens = my_tokenizer(review)
			tokens = word_tokenize(review)
			all_tokens.append(' '.join(tokens))
			for token in tokens:
				if token not in word2idx:
					word2idx[token] = current_index
					current_index += 1


	V = len(word2idx)
	max_len = max([len(x.split(' ')) for x in all_tokens])
	vocab_processor = learn.preprocessing.VocabularyProcessor(max_len)
	X = np.array(list(vocab_processor.fit_transform(all_tokens)))
	return X, y, V, vocab_processor


def fix_spaces(s):
	s = s.replace(' .', '.')
	s = s.replace(' ,', '.')
	s = s.replace(' !', '!')
	s = s.replace(' ?', '?')
	s = s.replace(' :', ':')
	s = s.replace(" '", "'")
	s = s.replace(' ;', ';')
	s = s.replace(' <', '<')
	s = s.replace('( ', '(')
	s = s.replace(' )', ')')
	s = s.replace(' >', '>')
	s = s.replace(' /', '/')
	s = s.replace('$ ', '$')
	s = s.replace(' & ', '&')
	s = s.replace('... ', '...')
	s = s.replace('< br/>', '')
	s = s.replace('< br/>', '')
	s = s.replace(" n't", "n't")
	return s


def get_char_data():

	dat = pd.read_csv('data/amazon-fine-foods/Reviews.csv', usecols=['Text'], nrows=1000)
	dat.Text = dat.Text.map(lambda x: x.decode('utf-8'))
	dat.Text = dat.Text.map(lambda x: word_tokenize(x))
	dat.Text = dat.Text.map(lambda x: x + ['\n'])
	dat.Text = dat.Text.map(lambda x: ' '.join(x))
	dat.Text = dat.Text.map(lambda x: fix_spaces(x))
	dat.Text = dat.Text.map(lambda x: x.encode('utf-8'))
	all_text = ' '.join(dat.Text.values)
	all_text = fix_spaces(all_text)
	print all_text[:100000]
	vocab = sorted(list(set(all_text)))
	vocab_size = len(vocab)
	idx2vocab = dict(enumerate(vocab))
	vocab2idx = dict(zip(idx2vocab.values(), idx2vocab.keys()))
	X = [vocab2idx[i] for i in all_text]
	return vocab, vocab_size, vocab2idx, idx2vocab, X








