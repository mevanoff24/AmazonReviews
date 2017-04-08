import numpy as np 
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os

from utils import get_data, bag_of_words_data, oneHot

def init_weight_and_bias(insize, outsize):
	w = np.random.randn(insize, outsize) / np.sqrt(insize + outsize)
	b = np.zeros(outsize)
	return w.astype(np.float32), b.astype(np.float32)


def batch_norm_wrapper(inputs, is_training, decay=0.999, epsilon=1e-3):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)


class HiddenLayer(object):
	def __init__(self, insize, outsize, count, batch_norm=False):
		self.insize = insize
		self.outsize = outsize
		W, b = init_weight_and_bias(insize, outsize)
		self.W = tf.Variable(W, name='W'+str(count))
		self.b = tf.Variable(b, name='b'+str(count))
		self.params = [self.W, self.b]

	def forward(self, X):
		if batch_norm:
			Z = tf.matmul(X, self.W) + self.b
			Z = batch_norm_wrapper(Z, is_training=True)
			return tf.nn.relu(Z)
		return tf.nn.relu(tf.matmul(X, self.W) + self.b)


class ANN(object):
	def __init__(self, hidden_layer_sizes, save_dir='nueralNetwork'):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.save_dir = save_dir

	def fit(self, X, y, learning_rate=1e-05, reg=0.1, batch_size=100, epochs=100, print_period=10, show_fig=False):

		X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.10, random_state=100)

		N, D = X_train.shape
		K = len(set(y))
		y_trainOh = oneHot(y_train)
		y_devOh = oneHot(y_dev)

		tfX = tf.placeholder(tf.float32, shape=[None, D], name='tfX')
		tfY = tf.placeholder(tf.float32, shape=[None, K], name='tfY')

		self.hidden_layers = []

		insize = D
		count = 0
		for outsize in self.hidden_layer_sizes:
			h = HiddenLayer(insize, outsize, count)
			self.hidden_layers.append(h)
			insize = outsize
			count += 1

		with tf.variable_scope('softmax'):
			W, b = init_weight_and_bias(insize, K)
			self.W = tf.Variable(W, name='W-soft')
			self.b = tf.Variable(b, name='b-soft')
			self.params = [self.W, self.b]

		for h in self.hidden_layers:
			self.params += h.params

		logits = self.forward(tfX)

		rcost = sum([tf.nn.l2_loss(p) for p in self.params])
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tfY, logits=logits)) + rcost * reg

		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		pred_opt = self.predict(tfX, train_phase=True)

		n_batches = N / batch_size
		train_costs = []
		dev_costs = []
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		with tf.Session() as sess:
			sess.run(init)
			for epoch in range(epochs):
				X_train, y_trainOh = shuffle(X_train, y_trainOh)
				for i in range(n_batches):
					batch_x = X_train[i * batch_size : (i * batch_size + batch_size)]
					batch_y = y_trainOh[i * batch_size : (i * batch_size + batch_size)]

					_, train_loss = sess.run([optimizer, loss], feed_dict={tfX: batch_x, tfY: batch_y})

					if i % print_period == 0:
						p = sess.run(pred_opt, feed_dict={tfX:X_dev})
						dev_loss = sess.run(loss, feed_dict={tfX:X_dev, tfY: y_devOh})
						acc = self.score(y_dev, p)
						train_costs.append(train_loss)
						dev_costs.append(dev_loss)
						print 'iteration:', i, 'epoch:', epoch, 'train_loss:', train_loss, 'dev_loss:', dev_loss, 'accuracy:', acc
			
			if not os.path.exists(self.save_dir):
				os.makedirs(self.save_dir)
			saver.save(sess, os.path.join(self.save_dir + '/model'))

		if show_fig:
			plt.plot(train_costs, label='train_costs')
			plt.plot(dev_costs, label='dev_costs')
			plt.legend()
			plt.show()

	def forward(self, X):
		Z = X 
		for h in self.hidden_layers:
			Z = h.forward(Z)
		return tf.matmul(Z, self.W) + self.b

	def predict(self, X, train_phase=False):
		if train_phase:
			Z = self.forward(X)
			return tf.argmax(Z, 1)
		else:
			X = np.array(X, dtype=np.float32)
			with tf.Session() as sess:
				new_saver = tf.train.import_meta_graph(os.path.join(self.save_dir + '/model.meta'))
				new_saver.restore(sess, tf.train.latest_checkpoint(self.save_dir + '/'))
				Z = self.forward(X)
				return tf.argmax(Z, 1).eval()

	def score(self, y_true, y_pred):
		return np.mean(y_true == y_pred)


if __name__ == '__main__':

	X, y = bag_of_words_data()

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=100) 

	model = ANN([512, 1024, 512])
	model.fit(X_train, y_train, learning_rate=1e-04, epochs=10, show_fig=True)

	preds = model.predict(X_test)
	print 'Test Set Accuracy', model.score(preds, y_test)

	pred = model.predict([X_test[10]])
	print 'Test Set Accuracy', model.score(pred, y_test[10])





