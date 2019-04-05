import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import uuid

from tornasole_hook import TornasoleHook

run_id = str(uuid.uuid4())

rng = np.random

# Parameters
learning_rate = 0.00001
training_epochs = 5000


# Training Data
#train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
#                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
#train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
#                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
#n_samples = train_X.shape[0]

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

mu, sigma = 0, 2
diff, mult = 1, 5
batch_size = 32
clip_factor = 1
display_step = 50
with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
    # Fit all training data
    for epoch in range(training_epochs):
        #print( epoch )
        train_X = np.random.normal(mu, sigma, batch_size)
        
        train_Y = train_X * mult + diff
        #print( train_X, train_Y )
        #train_Y = np.clip(train_Y, (diff+mu)-(sigma*mult)/clip_factor, (diff+mu)+(sigma*mult)/clip_factor)
        #for (x, y) in zip(train_X, train_Y):
        sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})
        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c, _W, _b = sess.run([cost, W, b], feed_dict={X: train_X, Y:train_Y})
            print( "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", _W, "b=", _b )

    print( "Optimization Finished!" )
    training_cost, _W, _b = sess.run([cost, W, b], feed_dict={X: train_X, Y: train_Y})
    print( "Training cost=", training_cost, "W=", _W, "b=", _b )

