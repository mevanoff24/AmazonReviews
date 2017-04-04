import numpy as np 
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os

from utils import get_sequence_data, oneHot

# def accuracy(y_true, y_pred):
#     return np.mean(y_true, y_pred)


def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)
    raw_x = raw_x[:, 0]

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)


def gen_batch_test(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)
    raw_x = raw_x[:, 0]

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps
    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        return x, y


def gen_epochs(n, batch_size, num_steps):
    for i in range(n):
        X, y, _, _ = get_sequence_data()
        yield gen_batch((X, y), batch_size, num_steps)


def length(data):
    relevant = tf.sign(tf.abs(data))
    length = tf.reduce_sum(relevant, axis=1)
    length = tf.cast(length, tf.int32)
    return length



class RNN(object):
    def __init__(self, M, V, save_dir='RNN'):
        self.M = M
        self.V = V
        self.save_dir = save_dir

    def fit(self, X, y, num_steps=10, num_layers=1, learning_rate=1e-05, reg=0.1, batch_size=32, epochs=100, print_period=10, show_fig=False):

        X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=500, random_state=100)

        N, D = X_train.shape
        K = len(set(y))
        y_trainOh = oneHot(y_train)
        y_devOh = oneHot(y_dev)

        tfX = tf.placeholder(tf.int32, shape=[None, num_steps], name='tfX')
        tfY = tf.placeholder(tf.int32, shape=[None, num_steps], name='tfY')
        tfY_OH = tf.one_hot(tfY, K)

        with tf.variable_scope('embedding-layer'):
            embeddings = tf.get_variable('embeddings', [self.V, self.M])
            rnn_inputs = tf.nn.embedding_lookup(embeddings, tfX, name='lookup')

        cell = tf.contrib.rnn.GRUCell(self.M)
        cell = tf.contrib.rnn.MultiRNNCell([cell]*num_layers)
        init_state = cell.zero_state(batch_size, dtype=tf.float32)

        seq_length = length(tfX)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, 
                                        initial_state=init_state, sequence_length=seq_length)
        rnn_outputs = tf.reshape(rnn_outputs, [-1, self.M])

        with tf.variable_scope('softmax-layer'):
            W = tf.get_variable('W', shape=[self.M, K])
            b = tf.get_variable('b', shape=[K], initializer=tf.constant_initializer(0.0))

        logits = tf.matmul(rnn_outputs, W) + b
        y_reshape = tf.reshape(tfY_OH, [batch_size*num_steps, -1])
        predictions = tf.nn.softmax(logits)

        batch_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_reshape, logits=logits)
        total_loss = tf.reduce_mean(batch_loss)
        loss_summary = tf.summary.scalar('loss', total_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
        pred_opt = tf.argmax(predictions, 1)
        pred_opt = tf.reshape(pred_opt, [batch_size, -1])
        # correct_pred = 
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tfY, tf.cast(pred_opt, tf.int32)), tf.float32))

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        train_costs = []
        dev_costs = []

        with tf.Session() as sess:
            try:
                print("Trying to restore last checkpoint ...")
                last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=self.save_dir)
                saver.restore(sess, save_path=last_chk_path)
                print("Restored checkpoint from:", last_chk_path)
            except:
                print("Failed to restore checkpoint. Initializing variables instead.")  
                sess.run(init)

            istate = sess.run(init_state)
            for idx, epoch in enumerate(gen_epochs(epochs, batch_size, num_steps)):
                for step, (batch_x, batch_y) in enumerate(epoch):
                    # _, train_loss, state, train_acc = sess.run([optimizer, total_loss, final_state, accuracy], 
                                            # feed_dict={tfX: batch_x, tfY: batch_y})
                    feed_dict={tfX: batch_x, tfY: batch_y}
                    for i, v in enumerate(init_state):
                        feed_dict[v] = istate[i]

                    _, train_loss, state, train_acc = sess.run([optimizer, total_loss, final_state, accuracy], 
                                            feed_dict=feed_dict)

                    if step % print_period == 0:
                        X_dev_batch, y_dev_batch = gen_batch_test((X_dev, y_dev), batch_size, num_steps)
                        dev_loss = sess.run(total_loss, feed_dict={tfX: X_dev_batch, tfY: y_dev_batch})
                        train_costs.append(train_loss)
                        dev_costs.append(dev_loss)
                        val_acc = sess.run(accuracy, feed_dict={tfX: X_dev_batch, tfY: y_dev_batch})
                        print 'step', step, 'epoch', idx, 'dev_loss', dev_loss, 'train_loss', train_loss, 'train accuracy', train_acc, 'val accuracy', val_acc
                  
                    istate = state

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            saver.save(sess, os.path.join(self.save_dir + '/model'))

        if show_fig:
            plt.plot(train_costs, label='train_costs')
            plt.plot(dev_costs, label='dev_costs')
            plt.legend()
            plt.show()
 
        



def main():
    X, y, V, vocab_processor = get_sequence_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=100) 

    model = RNN(512, V)
    model.fit(X_train, y_train, show_fig=True)


if __name__ == '__main__':
    main()

