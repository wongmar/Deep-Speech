import tensorflow as tf
from tensorflow.python.ops.constant_op import constant
#from tensorflow.nn.rnn_ import rnn, rnn_cell
from tensorflow.python.ops import functional_ops
import numpy as np
import sys

import import_data

def charToInt(c):
	if c == ' ':
		return 26
	else:
		return ord(c) - ord('a')


#Engine
t_flag = 1 #training flag
learning_rate = 0.0001
training_iters = 100
batch_size = 2
display_step = 10
dropout_rate = 0.05
relu_clip = 20
n_steps = 500
n_input = 160
n_context = 0 #Since we are already creating overlapping spectrograms, this does not need to be set.

n_hidden_1 = n_input + 2*n_input*n_context 
n_hidden_2 = n_input + 2*n_input*n_context 
n_hidden_5 = n_input + 2*n_input*n_context

n_cell_dim = n_input + 2*n_input*n_context
n_hidden_3 = 2 * n_cell_dim

n_character = 28

n_hidden_6 = n_character

x = tf.placeholder("float", [None, n_steps, n_input + 2*n_input*n_context])
y = tf.placeholder("string", [None,1])
#y = tf.placeholder("int", [None,n_steps])
z = tf.placeholder(tf.int32, [None, n_steps])

istate_fw = tf.placeholder("float", [None, 2*n_cell_dim])
istate_bw = tf.placeholder("float", [None, 2*n_cell_dim])

keep_prob = tf.placeholder(tf.float32)
keep_prob = 0.05


weights = {
	'h1': tf.Variable(tf.random_normal([n_input + 2*n_input*n_context, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
	'h5': tf.Variable(tf.random_normal([(2 * n_cell_dim), n_hidden_5])),
	'h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6]))
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'b3': tf.Variable(tf.random_normal([n_hidden_3])),
	'b5': tf.Variable(tf.random_normal([n_hidden_5])),
	'b6': tf.Variable(tf.random_normal([n_hidden_6]))
}

def BiRNN(_X, _istate_fw, _istate_bw, _weights, _biases):
	# Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
	_X = tf.transpose(_X, [1, 0, 2])  # Permute n_steps and batch_size
	# Reshape to prepare input for first layer
	_X = tf.reshape(_X, [-1, n_input + 2*n_input*n_context]) # (n_steps*batch_size, n_input + 2*n_input*n_context)

	#Hidden layer with clipped RELU activation and dropout
	layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])), relu_clip)
	layer_1 = tf.nn.dropout(layer_1, keep_prob)
	#Hidden layer with clipped RELU activation and dropout
	layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])), relu_clip)
	layer_2 = tf.nn.dropout(layer_2, keep_prob)
	#Hidden layer with clipped RELU activation and dropout
	layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3'])), relu_clip)
	layer_3 = tf.nn.dropout(layer_3, keep_prob)

	# Define lstm cells with tensorflow
	# Forward direction cell
	lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_cell_dim, forget_bias=1.0)
	# Backward direction cell
	lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_cell_dim, forget_bias=1.0)

	# Split data because rnn cell needs a list of inputs for the BRNN inner loop
	layer_3 = tf.split(0, n_steps, layer_3)

	# Get lstm cell output
	#						outputs is a tuple, must change to tensor
	outputs = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, layer_3, initial_state_fw=_istate_fw, initial_state_bw=_istate_bw) 
	
	# Reshape outputs from a list of n_steps tensors each of shape [batch_size, 2*n_cell_dim]
	# to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
	outputs = tf.pack(outputs[0])
	outputs = tf.reshape(outputs, [-1, 2*n_cell_dim])
	#Hidden layer with clipped RELU activation and dropout
	layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, _weights['h5']), _biases['b5'])), relu_clip)
	layer_5 = tf.nn.dropout(layer_5, keep_prob)
	
	#Hidden layer with softmax function
	layer_6 = tf.nn.softmax(tf.add(tf.matmul(layer_5, _weights['h6']), _biases['b6']))

	# Reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
	# to a tensor of shape [batch_size, n_steps, n_hidden_6]
	layer_6 = tf.reshape(layer_6, [n_steps, batch_size, n_hidden_6])
	#layer_6 = tf.transpose(layer_6, [1, 0, 2])  # Permute n_steps and batch_size
	
	# Return layer_6
	return layer_6
	
pred = BiRNN(x, istate_fw, istate_bw, weights, biases)



#Training

def SimpleSparseTensorFrom(x):
  """Create a very simple SparseTensor with dimensions (batch, time).
  Args:
    x: a list of lists of type int
  Returns:
    x_ix and x_val, the indices and values of the SparseTensor<2>.
  """
  x_ix = []
  x_val = []
  for batch_i, batch in enumerate(x):
    for time, val in enumerate(batch):
      x_ix.append([batch_i, time])
      x_val.append(val)
  x_shape = [len(x), np.asarray(x_ix).max(0)[1]+1]
  x_ix = tf.constant(x_ix, tf.int64)
  x_val = tf.constant(x_val, tf.int32)
  x_shape = tf.constant(x_shape, tf.int64)

  return tf.SparseTensor(x_ix, x_val, x_shape)


#This equalizes the lengths of the labels
#I wanted this to use the labels placeholder but its saved as a tensor
mnist = import_data.read_data_sets("data", batch_size,n_steps)
labs = []
for a in mnist.labels:
	tmp = [charToInt(c) for c in a[0]]
	labs.append(tmp)
labs += [] * (batch_size - len(labs))
for i in xrange(len(labs)-1):
	labs[i] += [0] * (n_steps - len(labs[i]))

lab = SimpleSparseTensorFrom(labs)

'''
#An attempt to have the program not read the data set twice. Since a tensor cannot be modified or read outside of a session, this function failed to work
def create_label(z):
	labs = []
	for a in z:
		tmp = [charToInt(c) for c in a[0]]
		labs.append(tmp)

	for i in xrange(len(labs)):
		labs[i] += [0] * (n_steps - len(labs[i]))
	labs += [''] * (batch_size - len(labs))


	#lab = tf.SparseTensor(indices=index, values=new_labs, shape=sh)
	lab = SimpleSparseTensorFrom(labs)
	return lab
'''

# Define loss and optimizer
s_length = np.full(batch_size,n_steps, dtype=np.int64)

print("Defining Cost")
#cost =  tf.reduce_mean(tf.contrib.ctc.ctc_loss(inputs=pred, labels=lab, sequence_length=s_length))
cost =  tf.reduce_mean(tf.contrib.ctc.ctc_loss(inputs=pred, labels=lab, sequence_length=s_length))

print("Defining Optimizer")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,aggregation_method=2) # Adam Optimizer

# Evaluate model ~~~~~DECODE~~~~~
#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
print("Defining Correct Label")
correct_pred = tf.argmax(pred,1)

print "Defining Accuracy"
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
print("Initializing Variables")
# Initializing the variables
init = tf.initialize_all_variables()



# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	saver = tf.train.Saver(weights)
	step = 1
	#print("if")
	# Keep training until reach max iterations
	
	if t_flag == 1:
		print("Start Training")
		mnist = import_data.read_data_sets("data", batch_size,n_steps)
		while step * batch_size < training_iters:
			batch_xs, batch_ys = mnist.next_batch(batch_size)
			# Reshape data
			xs_shaped = batch_xs.copy()
			xs_shaped.resize(batch_size,n_steps, n_input + 2*n_input*n_context)			
			
			print "Running Optimizer"
			sess.run(optimizer, feed_dict={x: xs_shaped, y: batch_ys,
											istate_fw: np.zeros((batch_size, 2*n_cell_dim)),
											istate_bw: np.zeros((batch_size, 2*n_cell_dim))})
			if step % display_step == 0:
				# Calculate batch accuracy
				print "Evaluate Accuracy"
				acc = sess.run(accuracy, feed_dict={x: xs_shaped, y: batch_ys,
													istate_fw: np.zeros((batch_size, 2*n_cell_dim)),
													istate_bw: np.zeros((batch_size, 2*n_cell_dim))})
				# Calculate batch loss
				print "Evaluate Loss"
				loss = sess.run(cost, feed_dict={x: xs_shaped, y: batch_ys,
												 istate_fw: np.zeros((batch_size, 2*n_cell_dim)),
												 istate_bw: np.zeros((batch_size, 2*n_cell_dim))})
				print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
													", Training Accuracy= " + "{:.5f}".format(acc)
			step += 1
			print "Step Completed"
		print "Optimization Finished!"
		save_path = saver.save(sess, "weights.ckpt")
		print("Model saved in file: %s" % save_path)
	if t_flag == 0:
		mnist = import_data.read_data_sets("data",1,n_steps,training=False)
		prediction=tf.argmax(pred,1)
		print "predictions", prediction.eval(feed_dict={x: mnist.images, y:mnist.labels[0], istate_fw: np.zeros((batch_size, 2*n_cell_dim)),
													istate_bw: np.zeros((batch_size, 2*n_cell_dim))}, session=sess)
	sess.close()
