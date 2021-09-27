import numpy
import tensorflow as tf
import os
import os.path
import random
import math
import time
from PIL import Image

BATCH_SIZE = 4
KERNEL_SIZE = 3

class Model:
	def _conv_layer(self, name, input_var, stride, in_channels, out_channels, options = {}):
		activation = options.get('activation', 'relu')
		dropout = options.get('dropout', None)
		padding = options.get('padding', 'SAME')
		batchnorm = options.get('batchnorm', False)
		transpose = options.get('transpose', False)

		with tf.variable_scope(name) as scope:
			if not transpose:
				filter_shape = [KERNEL_SIZE, KERNEL_SIZE, in_channels, out_channels]
			else:
				filter_shape = [KERNEL_SIZE, KERNEL_SIZE, out_channels, in_channels]
			kernel = tf.get_variable(
				'weights',
				shape=filter_shape,
				initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / KERNEL_SIZE / KERNEL_SIZE / in_channels)),
				dtype=tf.float32
			)
			biases = tf.get_variable(
				'biases',
				shape=[out_channels],
				initializer=tf.constant_initializer(0.0),
				dtype=tf.float32
			)
			if not transpose:
				output = tf.nn.bias_add(
					tf.nn.conv2d(
						input_var,
						kernel,
						[1, stride, stride, 1],
						padding=padding
					),
					biases
				)
			else:
				batch = tf.shape(input_var)[0]
				side = tf.shape(input_var)[1]
				output = tf.nn.bias_add(
					tf.nn.conv2d_transpose(
						input_var,
						kernel,
						[batch, side * stride, side * stride, out_channels],
						[1, stride, stride, 1],
						padding=padding
					),
					biases
				)
			if batchnorm:
				output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=self.is_training, decay=0.99)
			if dropout is not None:
				output = tf.nn.dropout(output, keep_prob=1-dropout)

			if activation == 'relu':
				return tf.nn.relu(output, name=scope.name)
			elif activation == 'sigmoid':
				return tf.nn.sigmoid(output, name=scope.name)
			elif activation == 'none':
				return output
			else:
				raise Exception('invalid activation {} specified'.format(activation))

	def _fc_layer(self, name, input_var, input_size, output_size, options = {}):
		activation = options.get('activation', 'relu')
		dropout = options.get('dropout', None)
		batchnorm = options.get('batchnorm', False)

		with tf.variable_scope(name) as scope:
			weights = tf.get_variable(
				'weights',
				shape=[input_size, output_size],
				initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / input_size)),
				dtype=tf.float32
			)
			biases = tf.get_variable(
				'biases',
				shape=[output_size],
				initializer=tf.constant_initializer(0.0),
				dtype=tf.float32
			)
			output = tf.matmul(input_var, weights) + biases
			if batchnorm:
				output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=self.is_training, decay=0.99)
			if dropout is not None:
				output = tf.nn.dropout(output, keep_prob=1-dropout)

			if activation == 'relu':
				return tf.nn.relu(output, name=scope.name)
			elif activation == 'sigmoid':
				return tf.nn.sigmoid(output, name=scope.name)
			elif activation == 'none':
				return output
			else:
				raise Exception('invalid activation {} specified'.format(activation))

	def __init__(self, bn=False, size=(512, 512), input_channels=6):
		tf.reset_default_graph()

		self.is_training = tf.placeholder(tf.bool)
		self.inputs = tf.placeholder(tf.uint8, [None, size[0], size[1], input_channels])
		self.float_inputs = tf.cast(self.inputs, tf.float32)/255.0# + tf.random.normal(tf.shape(self.inputs), stddev=0.04)*tf.cast(self.is_training, tf.float32)
		self.targets = tf.placeholder(tf.float32, [None, size[0]/4, size[1]/4])
		self.masks = tf.placeholder(tf.float32, [None, size[0]/4, size[1]/4])
		self.learning_rate = tf.placeholder(tf.float32)

		# layers
		self.layer1 = self._conv_layer('layer1', self.float_inputs, 2, input_channels, 32, {'batchnorm': False}) # -> 256x256x32
		self.layer2 = self._conv_layer('layer2', self.layer1, 2, 32, 64, {'batchnorm': bn}) # -> 128x128x64
		self.layer3 = self._conv_layer('layer3', self.layer2, 2, 64, 64, {'batchnorm': bn}) # -> 64x64x64
		self.layer4 = self._conv_layer('layer4', self.layer3, 2, 64, 64, {'batchnorm': bn}) # -> 32x32x64
		self.layer5 = self._conv_layer('layer5', self.layer4, 1, 64, 64, {'batchnorm': bn}) # -> 32x32x64
		self.layer6 = self._conv_layer('layer6', self.layer5, 2, 64, 64, {'batchnorm': bn, 'transpose': True}) # -> 64x64x64
		self.layer7 = self._conv_layer('layer7', self.layer6, 2, 64, 64, {'batchnorm': bn, 'transpose': True}) # -> 128x128x64
		#self.layer7 = tf.concat([self.layer2, tf.image.resize(self.layer5, [128, 128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)], axis=3)
		self.pre_outputs = self._conv_layer('pre_outputs', self.layer7, 1, 64, 1, {'activation': 'none', 'batchnorm': False})[:, :, :, 0] # -> 128x128x1
		self.outputs = tf.nn.sigmoid(self.pre_outputs)
		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets, logits=self.pre_outputs) * self.masks)

		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		self.init_op = tf.initialize_all_variables()
		self.saver = tf.train.Saver(max_to_keep=None)
