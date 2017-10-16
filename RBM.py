# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:17:52 2017

@author:  Yaser M. Taheri
"""
import tensorflow as tf
import numpy as np


class RBM:
    
      def __init__(self,visible_size, hidden_size, inp, learning_rate):
          self.visible_size = visible_size
          self.hidden_size = hidden_size
          self.Xt = inp
          self.learning_rate = learning_rate
        
          with tf.variable_scope('layer'):

               self.W = tf.get_variable(name="W", shape= [self.visible_size, self.hidden_size], initializer = tf.random_normal_initializer(stddev=0.01)) #The weight matrix that stores the edge weights
               self.b = tf.get_variable(name= "hidde_bias", shape=[1,self.hidden_size], initializer = tf.zeros_initializer(), )                     #The bias vector for the hidden layer
               self.c = tf.get_variable(name= "input_bias", shape=[1,self.visible_size], initializer = tf.zeros_initializer()) 

          
          def sample(p):
              
              prop = tf.random_uniform(tf.shape(p), 0,1)
              return tf.floor(prop + p) 
         
          self.h_prop = tf.nn.sigmoid(tf.matmul(self.Xt , self.W) + self.b)
          self.h_k = sample(self.h_prop)

          self.x_prop = tf.nn.sigmoid(tf.matmul(self.h_k , tf.transpose(self.W)) + self.c)
          self.x_k = sample(self.x_prop)
          
          self.hs_prop = tf.nn.sigmoid(tf.matmul(self.x_k , self.W) + self.b)
          self.hs_k = sample(self.hs_prop)
          
          
          self.W_del =  learning_rate * (1/10)*tf.subtract(tf.matmul(tf.transpose(self.Xt), self.h_k) , tf.matmul(tf.transpose(self.x_k), self.hs_k) )
          self.b_del =  learning_rate * (1/10)*tf.reduce_sum((tf.subtract(self.h_k , self.hs_k)), 0, keep_dims=True)
          self.c_del =  learning_rate * (1/10)*tf.reduce_sum(tf.subtract(self.Xt , self.x_k), 0, keep_dims=True)
          
          self.W_up = self.W.assign_add(self.W_del)
          self.b_up = self.b.assign_add(self.b_del)
          self.c_up = self.c.assign_add(self.c_del)
          
