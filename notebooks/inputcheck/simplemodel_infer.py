import numpy as np
import tensorflow as tf
import uuid

from tornasole_hook import TornasoleHook


# Let's load a previously saved meta graph in the default graph
# This function returns a Saver
saver = tf.train.import_meta_graph('model/model.meta')

# We can now access the default graph where all our metadata has been loaded
graph = tf.get_default_graph()
prediction = graph.get_tensor_by_name("pred:0")
product = graph.get_tensor_by_name("product:0")
cost = graph.get_tensor_by_name("cost:0")

run_id = str(uuid.uuid4())

#infer_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
#                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
tornasole_hook = TornasoleHook("./ts_outputs/infer/",include=["^product:0$", "^X:0$", "^pred:0$"],exclude=["Y"],
                    step_interval=1,
                    run_id=run_id, 
                    dry_run=False, single_file=False, 
                    local_reductions=False)

with tf.Session() as sess:
    # To initialize values with saved data
    saver.restore(sess, 'model/model')
    tornasole_hook.begin()
    for i in range(100):
        infer_X = np.random.uniform(0, 10)
        sess_args = tornasole_hook.before_run(None)
        #print( "SA=", sess_args.feed_dict)
        results = sess.run( sess_args.fetches, feed_dict={"X:0" : infer_X } )
        tornasole_hook.after_run(None,results)
        #print( _pred, _prod )
    tornasole_hook.end(sess)