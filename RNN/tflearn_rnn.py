from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences

from utils import get_sequence_data
from sklearn.model_selection import train_test_split


X, y, V, vocab_processor = get_sequence_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=100)

max_len = 751
n_classes = 3

# Data preprocessing
# Sequence padding
X_train = pad_sequences(X_train, maxlen=max_len, value=0.)
X_test = pad_sequences(X_test, maxlen=max_len, value=0.)
# Converting labels to binary vectors
y_train = to_categorical(y_train, nb_classes=n_classes)
y_test = to_categorical(y_test, nb_classes=n_classes)

# Network building
net = tflearn.input_data([None, max_len])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, n_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X_train, y_train, validation_set=(X_test, y_test), show_metric=True, batch_size=32, snapshot_step=500)

