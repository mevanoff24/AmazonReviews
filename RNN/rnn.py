import numpy as np 
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os

from utils import get_sequence_data, oneHot

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


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

    def fit(self, X, y, num_steps=1, num_layers=1, learning_rate=1e-05, reg=0.1, batch_size=32, epochs=100, print_period=10, show_fig=False):

        X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=500, random_state=100)

        N, D = X_train.shape
        K = len(set(y))
        y_trainOh = oneHot(y_train)
        y_devOh = oneHot(y_dev)

        tfX = tf.placeholder(tf.int32, shape=[None, D, num_steps], name='tfX')
        tfY = tf.placeholder(tf.int32, shape=[None, K], name='tfY')

        with tf.variable_scope('embedding-layer'):
            embeddings = tf.get_variable('embeddings', [self.V, self.M])
            rnn_inputs = tf.nn.embedding_lookup(embeddings, tfX, name='lookup')

        rnn_inputs = tf.reshape(rnn_inputs, (-1, D * num_steps, self.M))

        cell = tf.contrib.rnn.LSTMCell(self.M)

        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, dtype=tf.float32)
        rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
        rnn_outputs = tf.gather(rnn_outputs, int(rnn_outputs.get_shape()[0]) - 1)

        with tf.variable_scope('softmax-layer'):
            W = tf.get_variable('W', shape=[self.M, K])
            b = tf.get_variable('b', shape=[K], initializer=tf.constant_initializer(0.0))

        logits = tf.matmul(rnn_outputs, W) + b
        print logits.get_shape()

        predictions = tf.nn.softmax(logits)
        pred_opt = tf.argmax(predictions, 1)

        print tfY.get_shape()
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tfY, logits=logits))

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        n_batches = N // batch_size

        with tf.Session() as sess:
            try:
                print("Trying to restore last checkpoint ...")
                last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=self.save_dir)
                saver.restore(sess, save_path=last_chk_path)
                print("Restored checkpoint from:", last_chk_path)
            except:
                print("Failed to restore checkpoint. Initializing variables instead.")  
                sess.run(init)
            
            for epoch in range(epochs):
                X_train, y_train = shuffle(X_train, y_train)
                
                total_cost = 0.
                for i in range(n_batches):
                    batch_x = X_train[i * batch_size : (i * batch_size + batch_size)]
                    batch_x = batch_x.reshape(batch_size, D, num_steps)
                    batch_y = y_trainOh[i * batch_size : (i * batch_size + batch_size)]
                    
                    _, cost, pred_i = sess.run([optimizer, loss, pred_opt], feed_dict={tfX: batch_x, tfY: batch_y})
        #             acc = accuracy(pred, np.argmax(batch_y, 1))
        #             print cost, acc
                    total_cost += cost
                    if i % print_period == 0: 
                        val_pred = sess.run(pred_opt, feed_dict={tfX: X_dev.reshape(-1, D, num_steps), tfY: y_devOh})
                        acc = accuracy(val_pred, y_dev)

                        print 'epoch', epoch, 'cost', total_cost, 'accuracy', acc

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
    print y[:10]
    print X[:10]
    print V
    model = RNN(512, V)
    model.fit(X_train, y_train, show_fig=True)


if __name__ == '__main__':
    main()

