import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#import mnist dataset consisting of 28x28 images
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)

"""
    Define constants
"""
#the number of time steps to unroll the lstm for BPTT
time_steps = 28
#the number of hidden LSTM units, i.e., the dimension of the output array
num_units = 128
#each lstm units takes one row of the input image at a time
n_input = 28
#learning rate for adam
learning_rate = 0.001
#mnist is meant to be classified in 10 classes(0-9).
n_classes = 10
#size of batch
batch_size = 128

checkpoint_dir = os.path.dirname(os.path.abspath(__file__)) + '\\mnist'


"""
    Define the graph
"""
g = tf.Graph()
with g.as_default():
    
    with tf.variable_scope("mnist"):
        #input placeholder to feed training images to the network
        #of [Batch Size, Sequence Length, Input Dimension]
        x = tf.placeholder(tf.float32,[None,time_steps,n_input],name = "in")
            
        #output label placeholder to feed classification labels
        #probs in size of [Batch Size, Output Dimension] 
        y = tf.placeholder(tf.float32,[None,n_classes], name = "out")
        
        with tf.variable_scope("fc"):
            #Trainable weights and biases of appropriate shape to compute the input to 
            #the softmax layer from the outputs of the LSTM to predict the class label
            out_weights = tf.get_variable("W",[num_units,n_classes],initializer=tf.contrib.layers.xavier_initializer())
            out_bias = tf.get_variable("b",[n_classes],initializer=tf.contrib.layers.xavier_initializer())
    
    # create an lstm cell consisting of the array of hidden states
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units,forget_bias=1,
                                        initializer=tf.contrib.layers.xavier_initializer())
    
    # create an rnn network whose cell (i.e., layer) is specified by lstm_cell
    # the dimension of the output is [batch_size, time_steps, num_units].
    outputs,state = tf.nn.dynamic_rnn(lstm_cell,x,dtype=tf.float32)
    
    #converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] 
    prediction = tf.matmul(outputs[:,-1,:],out_weights)+out_bias
    
    #loss_function
    tf.losses.softmax_cross_entropy(y, prediction)
    loss = tf.losses.softmax_cross_entropy(y, prediction)
    #optimization
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    # model evaluation
    correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    
    # mean of the number of correct predictions 
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    #initialize variables
    init=tf.global_variables_initializer()

"""
    Execution of the graph
"""


# launch a session
with tf.Session(graph=g) as sess:
    sess.run(init)
    # visualize the structure of the graph
    writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
    itr = 1
    check = 10
    epoch=200
    while (itr<epoch):
        batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size)
    
        batch_x=batch_x.reshape((batch_size,time_steps,n_input))
        
        # run the optimization, i.e., forward, backpropagation, and update using the Adam optimizer
        sess.run(opt, feed_dict={x: batch_x, y: batch_y})
    
        if (itr%check==0):
            #run the accuracy and loss parts of the graph by substituting the
            #values in feed_dict for the corresponding input and label values.
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
            
            print("For iter {0:d}".format(itr))
            print("Accuracy {0:.4f}".format(acc))
            print("Loss {0:.5f}".format(los))
            print("__________________")
            
            # save the graph, and variables
            saver = tf.train.Saver()
            saver.save(sess, checkpoint_dir + '\\checkpoints', global_step=itr)
            print("MNIST variables saved.")
    
        itr += 1
    
    # make an inference
    pred = sess.run(prediction,feed_dict={x: batch_x[0][np.newaxis,:]})
    
    Probs = np.exp(pred)/np.sum(np.exp(pred))
    
    print("Estimation is {0:d}".format(np.argmax(Probs)))
    
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(1,1,1)
    ax.imshow(batch_x[0])
    plt.show(True)
    
    writer.close() 

# create the graph to continue training
with tf.Session() as sess:
    meta_graph_file = tf.train.get_checkpoint_state(checkpoint_dir).model_checkpoint_path + '.meta'
    saver = tf.train.import_meta_graph(meta_graph_file)
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
    print("MNIST graph is created")
    
    