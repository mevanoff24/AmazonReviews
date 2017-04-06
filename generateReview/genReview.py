import tensorflow as tf 
import numpy as np
import os

from utils import get_char_data



def ptb_iterator(raw_data, batch_size, num_steps, epochs, steps_ahead=1):

	data = np.array(raw_data)
	data_len = data.shape[0]
	nb_batches = (data_len - 1) // (batch_size * num_steps)
	assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
	rounded_data_len = nb_batches * batch_size * num_steps
	xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * num_steps])
	ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * num_steps])

	for epoch in range(epochs):
		for batch in range(nb_batches):
			x = xdata[:, batch * num_steps:(batch + 1) * num_steps]
			y = ydata[:, batch * num_steps:(batch + 1) * num_steps]
			x = np.roll(x, -epoch, axis=0)  
			y = np.roll(y, -epoch, axis=0)
			yield x, y, epoch

class CharRNN(object):
	def __init__(self, M, V, save_dir='CharRNN'):
		self.M = M
		self.V = V
		self.save_dir = save_dir

	def fit(self, X, num_steps=10, num_layers=3, learning_rate=1e-05, batch_size=32, epochs=1, print_period=50, fit_model=True, gen_text=False):

		tfX = tf.placeholder(tf.int32, shape=[batch_size, num_steps], name='tfX') 
		tfY = tf.placeholder(tf.int32, shape=[batch_size, num_steps], name='tfY')

		tfXOH = tf.one_hot(tfX, self.V)
		tfYOH = tf.one_hot(tfY, self.V)

		cell = tf.contrib.rnn.GRUCell(self.M)
		cell = tf.contrib.rnn.MultiRNNCell([cell]*num_layers)
		init_state = cell.zero_state(batch_size, dtype=tf.float32)
		rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, tfXOH, initial_state=init_state)
		final_state = tf.identity(final_state, name='final_state')

		with tf.variable_scope('softmax-layer'):
			W = tf.get_variable('W', shape=[self.M, self.V])
			b = tf.get_variable('b', shape=[self.V], initializer=tf.constant_initializer(0.0))

		rnn_outputs = tf.reshape(rnn_outputs, [-1, self.M])
		logits = tf.matmul(rnn_outputs, W) + b
		y_reshape = tf.reshape(tfYOH, [batch_size*num_steps, self.V])

		predictions = tf.nn.softmax(logits, name='predictions')
		Y_arg = tf.argmax(predictions, 1)
		Y_arg = tf.reshape(Y_arg, [batch_size, num_steps])

		batch_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_reshape, logits=logits)
		total_loss = tf.reduce_mean(batch_loss)

		global_step = tf.Variable(0.0, trainable=False)
		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

		accuracy = tf.reduce_mean(tf.cast(tf.equal(tfY, tf.cast(Y_arg, tf.int32)), tf.float32))

		loss_summary = tf.summary.scalar('loss', total_loss)
		accuracy_summary = tf.summary.scalar('accuracy', accuracy)
		summaries = tf.summary.merge([loss_summary, accuracy_summary])

		init = tf.global_variables_initializer()

		saver = tf.train.Saver()

		if gen_text:
			return dict(
						tfX = tfX,
						tfY = tfY,
						init_state = init_state,
						final_state = final_state,
						total_loss = total_loss,
						optimizer = optimizer,
						preds = predictions,
						saver = saver
					)

		if fit_model: 
			with tf.Session() as sess:
				summary_writer = tf.summary.FileWriter(os.path.join(self.save_dir + '/logs'), sess.graph)
				try:
					print("Trying to restore last checkpoint ...")
					last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=self.save_dir)
					saver.restore(sess, save_path=last_chk_path)
					print("Restored checkpoint from:", last_chk_path)
				except:
					print("Failed to restore checkpoint. Initializing variables instead.")  
					sess.run(init)

				# ostate = sess.run(init_state)
				for batch_x, batch_y, epoch in ptb_iterator(X, batch_size, num_steps, epochs):
					training_state = None

					feed_dict = {tfX: batch_x, tfY: batch_y}
					if training_state is not None:
						feed_dict[init_state] = training_state

					# for i, v in enumerate(init_state):
					# 	feed_dict[v] = ostate[i]
					
					# i_global, _, train_loss, isstate, sm = sess.run([global_step, optimizer, total_loss, final_state, summaries], 
													# feed_dict=feed_dict)
					i_global, _, train_loss, training_state, sm = sess.run([global_step, optimizer, total_loss, final_state, summaries], 
													feed_dict=feed_dict)
					summary_writer.add_summary(sm, i_global)


					if i_global % print_period == 0:
						feed_dict = {tfX: batch_x, tfY: batch_y}
						# for i, v in enumerate(init_state):
						# 	feed_dict[v] = ostate[i]

						acc, l = sess.run([accuracy, total_loss], feed_dict=feed_dict)
						print 'epoch', epoch, 'iteration', i_global, 'accuracy', acc, 'loss', l 

					# isstate = ostate

				if not os.path.exists(self.save_dir):
					os.makedirs(self.save_dir)
				saver.save(sess, os.path.join(self.save_dir + '/model'), global_step=global_step)



	def generate_text(self, X, num_chars, vocab2idx, idx2vocab, vocab_size, batch_size=1, num_steps=1, prompt='A', pick_top_chars=None):

		g = self.fit(X, batch_size=batch_size, num_steps=num_steps, fit_model=False, gen_text=True)
		tfX = g['tfX']
		tfY = g['tfY']
		init_state = g['init_state']
		final_state = g['final_state']
		total_loss = g['total_loss']
		optimizer = g['optimizer']
		preds = g['preds']
		saver = g['saver']
		with tf.Session() as sess:
			saver.restore(sess, tf.train.latest_checkpoint(self.save_dir + '/'))
			state = None
			current_char = vocab2idx[prompt]
			chars = [current_char]
			for i in range(num_chars):
				if state is not None:
					feed_dict = {tfX: [[current_char]]}
					for i, v in enumerate(init_state):
						feed_dict[v] = state[i]

				else:
					feed_dict = {tfX: [[current_char]]}
				p, state = sess.run([preds, final_state], feed_dict=feed_dict)

				if pick_top_chars is not None:
					pred = np.squeeze(p)
					pred[np.argsort(pred)[:-pick_top_chars]] = 0
					pred = pred / pred.sum()
					current_char = np.random.choice(vocab_size, 1, p=pred)[0]
				else:
					current_char = np.random.choice(vocab_size, 1, p=np.squeeze(preds))[0]
				
				chars.append(current_char)
		
		final = map(lambda x: idx2vocab[x], chars)
		print ''.join(final)



def main():

	vocab, vocab_size, vocab2idx, idx2vocab, X = get_char_data()

	model = CharRNN(512, vocab_size)
	model.fit(X)
	model.generate_text(vocab, 750, vocab2idx, idx2vocab, vocab_size, pick_top_chars=10)



if __name__ == '__main__':
	main()