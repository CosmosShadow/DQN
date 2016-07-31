# coding: utf-8
import numpy as np
import tensorflow as tf
import prettytensor as pt
import cv2

class DQN:
	def __init__(self,params,name):
		with tf.variable_scope(name):
			self.network_type = 'nips'
			self.params = params
			self.network_name = name

			self.x = tf.placeholder('float32',[None,84,84,4])
			self.q_t = tf.placeholder('float32',[None])
			self.actions = tf.placeholder("float32", [None, params['num_act']])
			self.rewards = tf.placeholder("float32", [None])
			self.terminals = tf.placeholder("float32", [None])

			x_pt = pt.wrap(self.x).sequential()
			x_pt.conv2d(8, 16, stride=4, edges='VALID', activation_fn=tf.nn.relu)
			x_pt.conv2d(4, 32, stride=3, edges='VALID', activation_fn=tf.nn.relu)
			x_pt.flatten()
			x_pt.fully_connected(256, activation_fn=tf.nn.relu)
			x_pt.fully_connected(params['num_act'])

			self.y = x_pt.as_layer()

			#Q,Cost,Optimizer
			self.discount = tf.constant(self.params['discount'])
			self.yj = tf.add(self.rewards, tf.mul(1.0-self.terminals, tf.mul(self.discount, self.q_t)))
			self.Qxa = tf.mul(self.y, self.actions)
			self.Q_pred = tf.reduce_max(self.Qxa, reduction_indices=1)

			self.cost = pt.wrap(self.Q_pred).l2_regression(self.yj)
			
			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			self.rmsprop = tf.train.RMSPropOptimizer(self.params['lr'],self.params['rms_decay'],0.0,self.params['rms_eps']).minimize(self.cost, global_step=self.global_step)


