# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:19:43 2017

@author: Yaser M. Taheri
"""

from RBM import RBM
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

num_epochs = 20000
Batch_size = 10
input_size = 28*28 
hidden_size = 64
learning_rate=0.1

tf.reset_default_graph()    

X = tf.placeholder(tf.float32, [None, input_size], name="x")

rbm = RBM(input_size,hidden_size,X,learning_rate) 


op = [rbm.W_up, rbm.c_up, rbm.b_up]



init = tf.global_variables_initializer()
#tf.default_graph().finalized()


with tf.Session() as sess:
    
    sess.run(init)
    
    for i in range(num_epochs):
    
        weights=sess.run(op,feed_dict = {X: mnist.train.next_batch(Batch_size)[0]})
             
        print ("iteration  ", i)   



    def plotNNFilter(w):
        filters = 64
    #plt.figure(1, figsize=(20,20))
        n_columns =8
        n_rows = np.ceil(filters / n_columns) + 1
        for i in range(64):
            plt.figure(1)
            plt.subplot(n_rows, n_columns, i+1)
            plt.imshow(np.reshape(w[:,i], [28,28]), cmap="gray")        
            
    plotNNFilter (weights[0])
    
    
    ##################### test ##################################################
    sess = tf.Session()
    a = mnist.train.next_batch(1)[0]
    def sample(p):              
        prop = tf.random_uniform(tf.shape(p), 0,1)
        return tf.floor(prop + p) 
         
    h_prop = tf.nn.sigmoid(tf.matmul(a , weights[0]) + weights[2])
    h_k = sample(h_prop)
    
    x_prop = tf.nn.sigmoid(tf.matmul(h_k , tf.transpose(weights[0])) + weights[1])
    x_k = sample(x_prop)
    plt.figure(2)

    plt.imshow(np.reshape(a, [28,28]), cmap="gray")
    
    xx=sess.run(x_k)
    plt.figure(3)
    plt.imshow(np.reshape(xx, [28,28]), cmap="gray")
    
