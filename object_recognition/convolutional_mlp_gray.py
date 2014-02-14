"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd_gray import LogisticRegression, load_data, save_parameters
from mlp_gray import HiddenLayer

import var_dump
import csv
from pickle import pickle
from unpickle import unpickle

class LeNetConvPoolLayer(object):
	"""Pool Layer of a convolutional network """

	def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
		"""
		Allocate a LeNetConvPoolLayer with shared variable internal parameters.

		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input: theano.tensor.dtensor4
		:param input: symbolic image tensor, of shape image_shape

		:type filter_shape: tuple or list of length 4
		:param filter_shape: (number of filters, num input feature maps,
							  filter height,filter width)

		:type image_shape: tuple or list of length 4
		:param image_shape: (batch size, num input feature maps,
							 image height, image width)

		:type poolsize: tuple or list of length 2
		:param poolsize: the downsampling (pooling) factor (#rows,#cols)
		"""

		assert image_shape[1] == filter_shape[1]
		self.input = input

		# there are "num input feature maps * filter height * filter width"
		# inputs to each hidden unit
		fan_in = numpy.prod(filter_shape[1:])
		# each unit in the lower layer receives a gradient from:
		# "num output feature maps * filter height * filter width" /
		#   pooling size
		fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
				   numpy.prod(poolsize))
		# initialize weights with random weights
		W_bound = numpy.sqrt(6. / (fan_in + fan_out))
		self.W = theano.shared(numpy.asarray(
			rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
			dtype=theano.config.floatX),
							   borrow=True)

		# the bias is a 1D tensor -- one bias per output feature map
		b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)

		# convolve input feature maps with filters
		conv_out = conv.conv2d(input=input, filters=self.W,
				filter_shape=filter_shape, image_shape=image_shape)

		# downsample each feature map individually, using maxpooling
		pooled_out = downsample.max_pool_2d(input=conv_out,
											ds=poolsize, ignore_border=True)

		# add the bias term. Since the bias is a vector (1D array), we first
		# reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
		# thus be broadcasted across mini-batches and feature map
		# width & height
		#self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		self.output = T.nnet.softplus(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')) # soft-max

		# store parameters of this layer
		self.params = [self.W, self.b]


def evaluate_lenet5(learning_rate=0.1, learning_rate2=0.05, learning_rate3=0.01, n_epochs=200,
					dataset='cifar-10-batches-py',
					nkerns=[6, 16], batch_size=20, mode='train', amount='full'): # nkerns coule be ok with [10, 50]
	""" Demonstrates lenet on MNIST dataset

	:type learning_rate: float
	:param learning_rate: learning rate used (factor for the stochastic
						  gradient)

	:type n_epochs: int
	:param n_epochs: maximal number of epochs to run the optimizer

	:type dataset: string
	:param dataset: path to the dataset used for training /testing (MNIST here)

	:type nkerns: list of ints
	:param nkerns: number of kernels on each layer
	"""

	#learning_rate = theano.shared(value=learning_rate, borrow=True)

	rng = numpy.random.RandomState(23455)

	datasets = load_data(dataset, mode=mode, amount=amount)

	if mode == 'train':
		train_set_x, train_set_y = datasets[0]
		valid_set_x, valid_set_y = datasets[1]
	else:
		test_set_x, test_set_y = datasets[0]

	# compute number of minibatches for training, validation and testing
	if mode == 'train':
		n_train_batches = train_set_x.get_value(borrow=True).shape[0]
		n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
		n_train_batches /= batch_size
		n_valid_batches /= batch_size
	else:
		n_test_batches = test_set_x.get_value(borrow=True).shape[0]
		n_test_batches /= batch_size

	# allocate symbolic variables for the data
	index = T.lscalar()  # index to a [mini]batch
	x = T.matrix('x')   # the data is presented as rasterized images
	y = T.ivector('y')  # the labels are presented as 1D vector of
						# [int] labels

	ishape = (32, 32)  # this is the size of CIFIA-10 images (gray-scaled)

	######################
	# BUILD ACTUAL MODEL #
	######################
	print '... building the model'

	# Reshape matrix of rasterized images of shape (batch_size,32*32)
	# to a 4D tensor, compatible with our LeNetConvPoolLayer
	layer0_input = x.reshape((batch_size, 1, 32, 32))

	# Construct the first convolutional pooling layer:
	# filtering reduces the image size to (32-5+1,32-5+1)=(28,28)
	# maxpooling reduces this further to (28/2,28/2) = (14,14)
	# 4D output tensor is thus of shape (batch_size,nkerns[0],14,14)
	layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
			image_shape=(batch_size, 1, 32, 32),
			filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

	# Construct the second convolutional pooling layer
	# filtering reduces the image size to (14-5+1,14-5+1)=(10,10)
	# maxpooling reduces this further to (10/2,10/2) = (5,5)
	# 4D output tensor is thus of shape (nkerns[0],nkerns[1],5,5)
	layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
			image_shape=(batch_size, nkerns[0], 14, 14),
			filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

	# the HiddenLayer being fully-connected, it operates on 2D matrices of
	# shape (batch_size,num_pixels) (i.e matrix of rasterized images).
	# This will generate a matrix of shape (20,50*5*5) = (20,1250) <-??
	layer2_input = layer1.output.flatten(2)

	# construct a fully-connected sigmoidal layer
	layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 5 * 5,
						 n_out=500, activation=T.tanh)

	# classify the values of the fully-connected sigmoidal layer
	layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

	## load the saved parameters
	if mode == 'test':
		learned_params = unpickle('params/convolutional_mlp_gray.pkl')

	# the cost we minimize during training is the NLL of the model
	cost = layer3.negative_log_likelihood(y)

	# create a function to compute the mistakes that are made by the model
	if mode == 'test':
		test_model = theano.function([index], layer3.errors(y),
				givens={
					x: test_set_x[index * batch_size: (index + 1) * batch_size],
					y: test_set_y[index * batch_size: (index + 1) * batch_size]})
	else:
		validate_model = theano.function([index], layer3.errors(y),
				givens={
					x: valid_set_x[index * batch_size: (index + 1) * batch_size],
					y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

		check_label = theano.function(inputs=[index],
				outputs=layer3.y_pair(y),
					givens={
						x: train_set_x[index * batch_size: (index + 1) * batch_size],
						y: train_set_y[index * batch_size: (index + 1) * batch_size]})

	# create a function to get the labels predicted by the model
	if mode == 'test':
		get_test_labels = theano.function([index], layer3.y_pred,
				givens={
					x: test_set_x[index * batch_size: (index + 1) * batch_size],
					layer0.W: learned_params[0],
					layer0.b: learned_params[1],
					layer1.W: learned_params[2],
					layer1.b: learned_params[3],
					layer2.W: learned_params[4],
					layer2.b: learned_params[5],
					layer3.W: learned_params[6],
					layer3.b: learned_params[7]})


	if mode == 'train':
		# create a list of all model parameters to be fit by gradient descent
		params = layer3.params + layer2.params + layer1.params + layer0.params
	
		# create a list of gradients for all model parameters
		grads = T.grad(cost, params)

	# train_model is a function that updates the model parameters by
	# SGD Since this model has many parameters, it would be tedious to
	# manually create an update rule for each model parameter. We thus
	# create the updates list by automatically looping over all
	# (params[i],grads[i]) pairs.
	if mode == 'train':
		updates = []
		for param_i, grad_i in zip(params, grads):
			updates.append((param_i, param_i - learning_rate * grad_i))

		updates2 = []
		for param_i, grad_i in zip(params, grads):
			updates2.append((param_i, param_i - learning_rate2 * grad_i))

		updates3 = []
		for param_i, grad_i in zip(params, grads):
			updates3.append((param_i, param_i - learning_rate3 * grad_i))

	if mode == 'train':
		train_model = theano.function([index], cost, updates=updates,
			  givens={
				x: train_set_x[index * batch_size: (index + 1) * batch_size],
				y: train_set_y[index * batch_size: (index + 1) * batch_size]})
		
		train_model2 = theano.function([index], cost, updates=updates2,
			  givens={
				x: train_set_x[index * batch_size: (index + 1) * batch_size],
				y: train_set_y[index * batch_size: (index + 1) * batch_size]})

		train_model3 = theano.function([index], cost, updates=updates3,
			  givens={
				x: train_set_x[index * batch_size: (index + 1) * batch_size],
				y: train_set_y[index * batch_size: (index + 1) * batch_size]})

	###############
	# TRAIN MODEL #
	###############
	print '... training the model'
	# early-stopping parameters
	if mode == 'train':
		patience = 10000  # look as this many examples regardless
		patience_increase = 2  # wait this much longer when a new best is
							   # found
		improvement_threshold = 0.999  # a relative improvement of this much is
									   # considered significant
		validation_frequency = min(n_train_batches, patience / 2)
								  # go through this many
								  # minibatche before checking the network
								  # on the validation set; in this case we
								  # check every epoch

	start_time = time.clock()

	if mode == 'train':
		best_params = None
		best_validation_loss = numpy.inf
		best_iter = 0
		test_score = 0.
		done_looping = False
	else:
		done_looping = True

	epoch = 0

	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):

			iter = (epoch - 1) * n_train_batches + minibatch_index

			if iter % 100 == 0:
				print 'training @ iter = ', iter

			if epoch == 1:
				cost_ij = train_model(minibatch_index)
			elif this_validation_loss < 0.45 and this_validation_loss > 0.35:
				cost_ij = train_model2(minibatch_index)
			elif this_validation_loss < 0.35:
				cost_ij = train_model3(minibatch_index)
			else:
				cost_ij = train_model(minibatch_index)

			## check the contents of predictions occasionaly
			'''
			if iter % 100 == 0:
				[prediction, true_label] = check_label(minibatch_index)
				print 'prediction:'
				print prediction
				print 'true_label:'
				print true_label
			'''

			## save the parameters
			if mode == 'train':
				get_params = theano.function(inputs=[], outputs=[layer0.W, layer0.b, layer1.W, layer1.b, layer2.W, layer2.b, layer3.W, layer3.b])
				save_parameters(get_params(), 'convolutional_mlp_gray')


			if (iter + 1) % validation_frequency == 0:

				# compute zero-one loss on validation set
				validation_losses = [validate_model(i) for i
									 in xrange(n_valid_batches)]
				this_validation_loss = numpy.mean(validation_losses)
				print('epoch %i, minibatch %i/%i, validation error %f %%' % \
					  (epoch, minibatch_index + 1, n_train_batches, \
					   this_validation_loss * 100.))

				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:

					#improve patience if loss improvement is good enough
					if this_validation_loss < best_validation_loss *  \
					   improvement_threshold:
						patience = max(patience, iter * patience_increase)

					# save best validation score and iteration number
					best_validation_loss = this_validation_loss
					best_iter = iter

					'''
					# test it on the test set
					test_losses = [test_model(i) for i in xrange(n_test_batches)]
					test_score = numpy.mean(test_losses)
					print(('	 epoch %i, minibatch %i/%i, test error of best '
						   'model %f %%') %
						  (epoch, minibatch_index + 1, n_train_batches,
						   test_score * 100.))
					'''


			'''
			if patience <= iter:
				done_looping = True
				break
			'''


	if mode == 'test':
		print 'predicting the labels...'
		pred_labels = [[0 for j in xrange(batch_size)] for i in xrange(n_test_batches)]
		for i in xrange(n_test_batches):
			print str(i+1), '/', str(n_test_batches)
			pred_labels[i] = get_test_labels(i)

		writer = csv.writer(file('result/convolutional_mlp_gray.csv', 'w'))
		row = 1

		print 'output test labels...'
		for i in xrange(len(pred_labels)): # TBF: hard code
			print str(i+1), '/', str(len(pred_labels))
			for j in xrange(len(pred_labels[i])):
				writer.writerow([row, pred_labels[i][j]])
				row += 1


	end_time = time.clock()
	if mode == 'train':
		print('Optimization complete.')
		print('Best validation score of %f %% obtained at iteration %i,'\
			  'with test performance %f %%' %
			  (best_validation_loss * 100., best_iter + 1, test_score * 100.))
	print >> sys.stderr, ('The code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
	argvs = sys.argv
	if len(argvs) < 2:
		evaluate_lenet5()
	else:
		if argvs[1] == 'test':
			mode_ = 'test'
		else:
			mode_ = 'train'
		if len(argvs) > 2:
			if argvs[2] == 'min':
				amount_ = 'min'
			else:
				amount_ = 'full'
			evaluate_lenet5(mode=mode_,amount=amount_)
		else:
			evaluate_lenet5(mode=mode_)


def experiment(state, channel):
	evaluate_lenet5(state.learning_rate, dataset=state.dataset)
