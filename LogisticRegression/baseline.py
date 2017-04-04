import numpy as np 
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os

from utils import bag_of_words_data, oneHot



class LogisticRegression(object):
	def __init__(self, save_dir='LogisticRegression'):
		self.save_dir = save_dir

	def fit(self, X, y, learning_rate=1e-05, batch_size=100, epochs=100, print_period=10, show_fig=False):

		X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.10, random_state=100)

		N, D = X_train.shape
		K = len(set(y))
		y_trainOh = oneHot(y_train)
		y_devOh = oneHot(y_dev)

		tfX = tf.placeholder(tf.float32, shape=[None, D], name='tfX')
		tfY = tf.placeholder(tf.float32, shape=[None, K], name='tfY')

		with tf.variable_scope('variables'):
			self.W = tf.get_variable('W', shape=[D, K])
			self.b = tf.get_variable('b', shape=[K], initializer=tf.constant_initializer(0.0))

		logits = self.forward(tfX)

		batch_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tfY)
		total_loss = tf.reduce_mean(batch_loss)
		loss_summary = tf.summary.scalar('total_loss', total_loss)

		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
		pred_opt = self.predict(tfX, train_phase=True)

		n_batches = N / batch_size
		init = tf.global_variables_initializer()

		train_costs = []
		dev_costs = []
		summaries = tf.summary.merge([loss_summary])

		saver = tf.train.Saver()

		with tf.Session() as sess:
			writer = tf.summary.FileWriter(self.save_dir, sess.graph)
			sess.run(init)
			for epoch in range(epochs):
				X_train, y_trainOh = shuffle(X_train, y_trainOh)
				for i in range(n_batches):
					batch_x = X_train[i * batch_size : (i * batch_size + batch_size)]
					batch_y = y_trainOh[i * batch_size : (i * batch_size + batch_size)]

					_, l, smm = sess.run([optimizer, total_loss, summaries], feed_dict={tfX: batch_x, tfY: batch_y})
					writer.add_summary(smm, i)

					if i % print_period == 0:
						dev_loss, tsmm = sess.run([total_loss, summaries], feed_dict={tfX: X_dev, tfY: y_devOh})
						pred = sess.run(pred_opt, feed_dict={tfX: X_dev})
						acc = self.score(y_dev, pred)
						train_costs.append(l)
						dev_costs.append(dev_loss)
						writer.add_summary(tsmm, i)
						print 'iteration:', i, 'epoch:', epoch, 'train_loss:', l, 'dev_loss:', dev_loss, 'accuracy:', acc
			writer.close()

			if not os.path.exists(self.save_dir):
				os.makedirs(self.save_dir)
			saver.save(sess, os.path.join(self.save_dir + '/model'))
		
		if show_fig:
			plt.plot(train_costs, label='train_costs')
			plt.plot(dev_costs, label='dev_costs')
			plt.legend()
			plt.show()


	def forward(self, X):
		with tf.variable_scope('forward'):
			logits = tf.matmul(X, self.W) + self.b
		return logits

	def internal_predict(self, X):
		Z = self.forward(X)
		return tf.argmax(Z, 1)

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
	print X.shape

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=101, random_state=100) 

	model = LogisticRegression()
	model.fit(X_train, y_train, learning_rate=1e-04, epochs=100, show_fig=True)

	preds = model.predict(X_test)
	print 'Test Set Accuracy', model.score(preds, y_test)

	pred = model.predict([X_test[10]])
	print 'Test Set Accuracy', model.score(pred, y_test[10])








