import argparse
import numpy as np
import tensorflow as tf
import uuid

from collections import namedtuple
from tornasole_tf.hook import TornasoleHook

parser = argparse.ArgumentParser()
parser.add_argument('--mean', type=float, help='mean of the input distribution', required=True)
parser.add_argument('--stddev', type=float, help='stddev of the input distribution', required=True)
parser.add_argument('--batchsize', type=int, help='batch size', required=True)
parser.add_argument('--steps', type=int, help='number of inferences', required=True)
args = parser.parse_args()


#Temporary hack to make the results of inference look like training
packedresults = namedtuple('Res', ['results'])


# Let's load a previously saved meta graph in the default graph
# This function returns a Saver
saver = tf.train.import_meta_graph('model/model.meta')

# Define a run ID so that we can find this run in Montecarlo
run_id = str(uuid.uuid4())

# Create a hook that will run with the inference
tornasole_hook = TornasoleHook("./ts_outputs/infer/",
                    include=["^product:0$", "^X:0$", "^Y_hat:0$"],
                    exclude=[],
                    step_interval=1,
                    run_id=run_id, 
                    dry_run=False, 
                    single_file=True, 
                    local_reductions=False)

with tf.Session() as sess:
    # To initialize values with saved data
    saver.restore(sess, 'model/model')
    tornasole_hook.begin()
    for i in range(args.steps):
        infer_X = np.random.normal(args.mean, args.stddev, args.batchsize)
        sess_args = tornasole_hook.before_run(None)
        results = sess.run( sess_args.fetches, feed_dict={"X:0" : infer_X } )
        # Temporary hack
        results = packedresults(results)
        tornasole_hook.after_run(None,results)
        if i % (args.steps//10) == 0:
            print( f'Running sample {i}')
    tornasole_hook.end(sess)
