import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import uuid

from tornasole_tf.hook import TornasoleHook

parser = argparse.ArgumentParser()
parser.add_argument('--mean', type=float, help='mean of the input distribution', required=True)
parser.add_argument('--stddev', type=float, help='stddev of the input distribution', required=True)
parser.add_argument('--batchsize', type=int, help='batch size', required=True)
parser.add_argument('--epochs', type=int, help='number of epochs', required=True)
args = parser.parse_args()

run_id = str(uuid.uuid4())

# Parameters
learning_rate = 0.00001

# tf Graph Input
X = tf.placeholder("float", name="X")
Y = tf.placeholder("float", name="Y")

# Set model weights
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# Perform the multiplication
prod = tf.multiply(X, W, name="product")

# Construct a linear model
Y_hat = tf.add(prod, b, name="Y_hat")

# Mean squared error
cost = tf.reduce_sum(tf.pow(Y_hat-Y, 2), name="cost")  #/(2*n_samples)
#cost = tf.identity(cost, name="cost")

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

tornasole_hook = TornasoleHook("./ts_outputs/train/",include=[],exclude=[],
                    step_interval=10,
                    run_id=run_id, 
                    dry_run=False, single_file=True, 
                    local_reductions=False)

#
#    tf.train.Saver().
class SaveAtEnd(tf.train.SessionRunHook):
    def __init__(self, filename):
        self.filename = filename
    
    def begin(self):
        self._saver = tf.train.Saver()

    def end(self, session):
        self._saver.save(session, self.filename)


# Start training
hooks=[tornasole_hook, SaveAtEnd("./model/model")]
with tf.Session() as sess:
    sess.run(init)


diff, mult = 1, 5
with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
    # Fit all training data
    for epoch in range(args.epochs):
        train_X = np.random.normal(args.mean, args.stddev, args.batchsize)
        
        train_Y = train_X * mult + diff
        sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})
        #Display logs per epoch step
        if epoch % (args.epochs//10) == 0:
            c, _W, _b = sess.run([cost, W, b], feed_dict={X: train_X, Y:train_Y})
            print( "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", _W, "b=", _b )

    print( "Optimization Finished!" )
    training_cost, _W, _b = sess.run([cost, W, b], feed_dict={X: train_X, Y: train_Y})
    print( "Training cost=", training_cost, "W=", _W, "b=", _b )

