#!/usr/local/bin/python
"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
				&= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets, and a conjugate gradient optimization method
that is suitable for smaller datasets.


References:

	- textbooks: "Pattern Recognition and Machine Learning" -
				 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

import var_dump
import csv
from pickle import pickle
from unpickle import unpickle


class LogisticRegression(object):
	"""Multi-class Logistic Regression Class

	The logistic regression is fully described by a weight matrix :math:`W`
	and bias vector :math:`b`. Classification is done by projecting data
	points onto a set of hyperplanes, the distance to which is used to
	determine a class membership probability.
	"""

	def __init__(self, input, n_in, n_out):
		""" Initialize the parameters of the logistic regression

		:type input: theano.tensor.TensorType
		:param input: symbolic variable that describes the input of the
					  architecture (one minibatch)

		:type n_in: int
		:param n_in: number of input units, the dimension of the space in
					 which the datapoints lie

		:type n_out: int
		:param n_out: number of output units, the dimension of the space in
					  which the labels lie

		"""

		# initialize with 0 the weights W as a matrix of shape (n_in, n_out)
		self.W = theano.shared(value=numpy.zeros((n_in, n_out),
												 dtype=theano.config.floatX),
								name='W', borrow=True)
		# initialize the baises b as a vector of n_out 0s
		self.b = theano.shared(value=numpy.zeros((n_out,),
												 dtype=theano.config.floatX),
							   name='b', borrow=True)

		# compute vector of class-membership probabilities in symbolic form
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

		# compute prediction as class whose probability is maximal in
		# symbolic form
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)

		# parameters of the model
		self.params = [self.W, self.b]

	def negative_log_likelihood(self, y):
		"""Return the mean of the negative log-likelihood of the prediction
		of this model under a given target distribution.

		.. math::

			\frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
			\frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
				\ell (\theta=\{W,b\}, \mathcal{D})

		:type y: theano.tensor.TensorType
		:param y: corresponds to a vector that gives for each example the
				  correct label

		Note: we use the mean instead of the sum so that
			  the learning rate is less dependent on the batch size
		"""
		# y.shape[0] is (symbolically) the number of rows in y, i.e.,
		# number of examples (call it n) in the minibatch
		# T.arange(y.shape[0]) is a symbolic vector which will contain
		# [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
		# Log-Probabilities (call it LP) with one row per example and
		# one column per class LP[T.arange(y.shape[0]),y] is a vector
		# v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
		# LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
		# the mean (across minibatch examples) of the elements in v,
		# i.e., the mean log-likelihood across the minibatch.
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):
		"""Return a float representing the number of errors in the minibatch
		over the total number of examples of the minibatch ; zero one
		loss over the size of the minibatch

		:type y: theano.tensor.TensorType
		:param y: corresponds to a vector that gives for each example the
				  correct label
		"""

		# check if y has same dimension of y_pred
		if y.ndim != self.y_pred.ndim:
			raise TypeError('y should have the same shape as self.y_pred',
				('y', target.type, 'y_pred', self.y_pred.type))
		# check if y is of the correct datatype
		if y.dtype.startswith('int'):
			# the T.neq operator returns a vector of 0s and 1s, where 1
			# represents a mistake in prediction
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()

	def y_pair(self,y):
		pairs = [self.y_pred, y]
		return pairs


def load_data(dataset):
	''' Loads the dataset

	:type dataset: string
	:param dataset: the path to the dataset (here MNIST)
	'''

	#############
	# LOAD DATA #
	#############

	# Download the MNIST dataset if it is not present

	print '... loading data'


	## Load the dataset
	print 'min training...'
	train_set = unpickle('data/min_train_set_gray.pkl')
	valid_set = unpickle('data/min_valid_set_gray.pkl')
	print 'loading test data...'

	test_set = unpickle('data/test_set_gray_1.pkl')
	test_set = (test_set, [0 for i in xrange(0,len(test_set))])
	print 'done!'

	#train_set, valid_set, test_set format: tuple(input, target)
	#input is an numpy.ndarray of 2 dimensions (a matrix)
	#witch row's correspond to an example. target is a
	#numpy.ndarray of 1 dimensions (vector)) that have the same length as
	#the number of rows in the input. It should give the target
	#target to the example with the same index in the input.

	def shared_dataset(data_xy, borrow=True):
		""" Function that loads the dataset into shared variables

		The reason we store our dataset in shared variables is to allow
		Theano to copy it into the GPU memory (when code is run on GPU).
		Since copying data into the GPU is slow, copying a minibatch everytime
		is needed (the default behaviour if the data is not in a shared
		variable) would lead to a large decrease in performance.
		"""
		data_x, data_y = data_xy
		shared_x = theano.shared(numpy.asarray(data_x,
											   dtype=theano.config.floatX),
								 borrow=borrow)
		shared_y = theano.shared(numpy.asarray(data_y,
											   dtype=theano.config.floatX),
								 borrow=borrow)
		# When storing data on the GPU it has to be stored as floats
		# therefore we will store the labels as ``floatX`` as well
		# (``shared_y`` does exactly that). But during our computations
		# we need them as ints (we use labels as index, and if they are
		# floats it doesn't make sense) therefore instead of returning
		# ``shared_y`` we will have to cast it to int. This little hack
		# lets ous get around this issue
		return shared_x, T.cast(shared_y, 'int32')

	train_set_x, train_set_y = shared_dataset(train_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	test_set_x, test_set_y = shared_dataset(test_set)

	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

	return rval



'''
def load_data(dataset, mode='train', amount='full'):
	# Loads the dataset

	#:type dataset: string
	#:param dataset: the path to the dataset (here MNIST)

	#############
	# LOAD DATA #
	#############

	# Download the MNIST dataset if it is not present

	print '... loading data'


	## Load the dataset
	if mode == 'train':
		# load training and validation data
		if amount == 'full':
			print 'full training...'
			train_set = unpickle('data/train_set_gray.pkl')
			valid_set = unpickle('data/valid_set_gray.pkl')
		elif amount == 'min':
			print 'min training...'
			train_set = unpickle('data/min_train_set_gray.pkl')
			valid_set = unpickle('data/min_valid_set_gray.pkl')
		else:
			print 'amount shoule be either full or min'
			raise NotImplementedError()
	else:
		# load test data
		#test_set = unpickle('data/test_set_gray.pkl')
		print 'loading test data...'
		if amount == 'full':
			test_set = []
			for i in xrange(1, 301): # from 1 to 300 TBF: hard code
				print str(i), '/', str(300)
				test_set_batch = unpickle('data/test_set_gray_' + str(i) + '.pkl')
				test_set.extend(test_set_batch)
			test_set = (test_set, [0 for i in xrange(0,len(test_set))])
		else:
			test_set = unpickle('data/test_set_gray_1.pkl')
			test_set = (test_set, [0 for i in xrange(0,len(test_set))])
		print 'done!'

	#train_set, valid_set, test_set format: tuple(input, target)
	#input is an numpy.ndarray of 2 dimensions (a matrix)
	#witch row's correspond to an example. target is a
	#numpy.ndarray of 1 dimensions (vector)) that have the same length as
	#the number of rows in the input. It should give the target
	#target to the example with the same index in the input.

	def shared_dataset(data_xy, borrow=True):
		""" Function that loads the dataset into shared variables

		The reason we store our dataset in shared variables is to allow
		Theano to copy it into the GPU memory (when code is run on GPU).
		Since copying data into the GPU is slow, copying a minibatch everytime
		is needed (the default behaviour if the data is not in a shared
		variable) would lead to a large decrease in performance.
		"""
		data_x, data_y = data_xy
		shared_x = theano.shared(numpy.asarray(data_x,
											   dtype=theano.config.floatX),
								 borrow=borrow)
		shared_y = theano.shared(numpy.asarray(data_y,
											   dtype=theano.config.floatX),
								 borrow=borrow)
		# When storing data on the GPU it has to be stored as floats
		# therefore we will store the labels as ``floatX`` as well
		# (``shared_y`` does exactly that). But during our computations
		# we need them as ints (we use labels as index, and if they are
		# floats it doesn't make sense) therefore instead of returning
		# ``shared_y`` we will have to cast it to int. This little hack
		# lets ous get around this issue
		return shared_x, T.cast(shared_y, 'int32')

	if mode == 'train':
		train_set_x, train_set_y = shared_dataset(train_set)
		valid_set_x, valid_set_y = shared_dataset(valid_set)
	else:
		test_set_x, test_set_y = shared_dataset(test_set)

	if mode == 'train':
		rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
	else:
		rval = [(test_set_x, test_set_y)]

	return rval
'''



def save_parameters(params, file_name):
	pickle(params, 'params/' + file_name + '.pkl')

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
						   dataset='cifar-10-batches-py',
						   batch_size=1000, mode='train', amount='full'):

	"""
	Demonstrate stochastic gradient descent optimization of a log-linear
	model

	This is demonstrated on MNIST.

	:type learning_rate: float
	:param learning_rate: learning rate used (factor for the stochastic
						  gradient)

	:type n_epochs: int
	:param n_epochs: maximal number of epochs to run the optimizer

	:type dataset: string
	:param dataset: the path of the MNIST dataset file from
				 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

	"""
	datasets = load_data(dataset, mode=mode, amount=amount)

	if mode == 'train':
		train_set_x, train_set_y = datasets[0]
		valid_set_x, valid_set_y = datasets[1]
	else:
		test_set_x, test_set_y = datasets[0]

	# compute number of minibatches for training, validation and testing
	if mode == 'train':
		n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
		n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
	else:
		n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

	######################
	# BUILD ACTUAL MODEL #
	######################
	print '... building the model'

	# allocate symbolic variables for the data
	index = T.lscalar()  # index to a [mini]batch
	x = T.matrix('x')  # the data is presented as rasterized images
	y = T.ivector('y')  # the labels are presented as 1D vector of
						   # [int] labels

	# construct the logistic regression class
	# Each MNIST image has size 28*28
	classifier = LogisticRegression(input=x, n_in=32 * 32, n_out=10)


	## load the saved parameters
	if mode == 'test':
		learned_params = unpickle('params/logistic_sgd_gray.pkl')


	# the cost we minimize during training is the negative log likelihood of
	# the model in symbolic format
	cost = classifier.negative_log_likelihood(y)

	# compiling a Theano function that computes the mistakes that are made by
	# the model on a minibatch
	if mode == 'test':
		test_model = theano.function(inputs=[index],
				outputs=classifier.errors(y),
				givens={
					x: test_set_x[index * batch_size: (index + 1) * batch_size],
					y: test_set_y[index * batch_size: (index + 1) * batch_size]})
	else:
		validate_model = theano.function(inputs=[index],
				outputs=classifier.errors(y),
				givens={
					x: valid_set_x[index * batch_size:(index + 1) * batch_size],
					y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

		check_label = theano.function(inputs=[index],
				outputs=classifier.y_pair(y),
					givens={
						x: train_set_x[index * batch_size: (index + 1) * batch_size],
						y: train_set_y[index * batch_size: (index + 1) * batch_size]})

	# create a function to get the labels predicted by the model
	if mode == 'test':
		get_test_labels = theano.function([index], classifier.y_pred,
				givens={
					x: test_set_x[index * batch_size: (index + 1) * batch_size],
					classifier.W: learned_params[0],
					classifier.b: learned_params[1]})

	# compute the gradient of cost with respect to theta = (W,b)
	if mode == 'train':
		g_W = T.grad(cost=cost, wrt=classifier.W)
		g_b = T.grad(cost=cost, wrt=classifier.b)

	# specify how to update the parameters of the model as a list of
	# (variable, update expression) pairs.
	if mode == 'train':
		updates = [(classifier.W, classifier.W - learning_rate * g_W),
				   (classifier.b, classifier.b - learning_rate * g_b)]

	# compiling a Theano function `train_model` that returns the cost, but in
	# the same time updates the parameter of the model based on the rules
	# defined in `updates`
	if mode == 'train':
		train_model = theano.function(inputs=[index],
				outputs=cost,
				updates=updates,
				givens={
					x: train_set_x[index * batch_size:(index + 1) * batch_size],
					y: train_set_y[index * batch_size:(index + 1) * batch_size]})


	###############
	# TRAIN MODEL #
	###############
	print '... training the model'
	# early-stopping parameters
	if mode == 'train':
		patience = 5000  # look as this many examples regardless
		patience_increase = 2  # wait this much longer when a new best is
									  # found
		improvement_threshold = 0.995  # a relative improvement of this much is
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
		test_score = 0.
		done_looping = False
	else:
		done_looping = True

	epoch = 0
	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):

			minibatch_avg_cost = train_model(minibatch_index)
			
			# iteration number
			iter = (epoch - 1) * n_train_batches + minibatch_index

			if (iter + 1) % validation_frequency == 0:
				# compute zero-one loss on validation set
				validation_losses = [validate_model(i)
									 for i in xrange(n_valid_batches)]
				this_validation_loss = numpy.mean(validation_losses)

				print('epoch %i, minibatch %i/%i, validation error %f %%' % \
					(epoch, minibatch_index + 1, n_train_batches,
					this_validation_loss * 100.))

				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:
					#improve patience if loss improvement is good enough
					if this_validation_loss < best_validation_loss *  \
					   improvement_threshold:
						patience = max(patience, iter * patience_increase)

					best_validation_loss = this_validation_loss
					'''
					# test it on the test set

					test_losses = [test_model(i)
								   for i in xrange(n_test_batches)]
					test_score = numpy.mean(test_losses)

					print(('	 epoch %i, minibatch %i/%i, test error of best'
					   ' model %f %%') %
						(epoch, minibatch_index + 1, n_train_batches,
						 test_score * 100.))
					'''


			if patience <= iter:
				done_looping = True
				break


	#[prediction, true_label] = check_label(minibatch_index)
	#print 'prediction:', prediction, 'true_label:', true_label

	# output test labels
	if mode == 'test':
		print 'predicting the labels...'
		pred_labels = [[0 for j in xrange(batch_size)] for i in xrange(n_test_batches)]
		for i in xrange(n_test_batches):
			print str(i+1), '/', str(n_test_batches)
			pred_labels[i] = get_test_labels(i)

		writer = csv.writer(file('result/logistic_sgd_gray.csv', 'w'))
		row = 1

		print 'output test labels...'
		for i in xrange(len(pred_labels)): # TBF: hard code
			print str(i+1), '/', str(len(pred_labels))
			for j in xrange(len(pred_labels[i])):
				writer.writerow([row, pred_labels[i][j]])
				row += 1


	end_time = time.clock()
	if mode == 'train':
		print(('Optimization complete with best validation score of %f %%,'
			   'with test performance %f %%') %
					 (best_validation_loss * 100., test_score * 100.))
		print 'The code run for %d epochs, with %f epochs/sec' % (
			epoch, 1. * epoch / (end_time - start_time))
	print >> sys.stderr, ('The code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.1fs' % ((end_time - start_time)))
	if mode == 'train':
		print 'saving the parameters learned...'
		get_params = theano.function(inputs=[], outputs=[classifier.W, classifier.b])
		save_parameters(get_params(), 'logistic_sgd_gray')


if __name__ == '__main__':
	argvs = sys.argv
	if len(argvs) < 2:
		sgd_optimization_mnist()
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
			sgd_optimization_mnist(mode=mode_,amount=amount_)
		else:
			sgd_optimization_mnist(mode=mode_)
