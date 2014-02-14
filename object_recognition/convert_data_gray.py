#!/usr/local/bin/python

import var_dump
from numpy import *

def pickle(data, file):
	import cPickle
	fo = open(file, 'w')
	cPickle.dump(data, fo, protocol=1)
	fo.close()


def unpickle(file):
	import cPickle
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict


## main
if __name__ == '__main__':
	## Load the dataset
	# load training data
	data_data = []
	data_labels = []
	path = 'cifar-10-batches-py'

	print 'loading original training data...'
	for i in range(1,5):
		data = unpickle(path + '/data_batch_' + str(i))
		data_data.extend(data['data'] / 255.)
		data_labels.extend(data['labels'])

	print 'converting to gray-scale...'
	train_set = (data_data, data_labels)
	train_set_gray = (zeros((len(train_set[0]),1024)), train_set[1])

	for i in range(0,len(train_set[0])):
		print '\r', round((i + 1) * 100 / len(train_set[0]), 0), '%',
		for j in range(0,1024):
			train_set_gray[0][i][j] = 0.21 * train_set[0][i][j] + 0.71 * train_set[0][i][1024*1+j] + 0.07 * train_set[0][i][1024*2+j]
	print 'done!'
	
	print 'dumping gray-scale training data into \'train_set_gray.pkl\'...'
	pickle(train_set_gray, 'data/train_set_gray.pkl')


	# load validation data
	print 'loading original validation data...'
	data = unpickle(path + '/data_batch_5')

	print 'converting to gray-scale...'
	valid_set = (data['data'] / 255., data['labels'])
	valid_set_gray = (zeros((len(valid_set[0]),1024)), valid_set[1])

	for i in range(0,len(valid_set[0])):
		print '\r', round((i + 1) * 100 / len(valid_set[0]), 0), '%',
		for j in range(0,1024):
			valid_set_gray[0][i][j] = 0.21 * valid_set[0][i][j] + 0.71 * valid_set[0][i][1024*1+j] + 0.07 * valid_set[0][i][1024*2+j]
	print 'done!'

	print 'dumping validation data into \'valid_set_gray.pkl\'...'
	pickle(valid_set_gray, 'data/valid_set_gray.pkl')
	

	# load test data
	print 'loading original test data...'
	data = unpickle(path + '/test_batch')

	print 'converting to gray-scale...'
	test_set = (data['data'] / 255., data['labels'])
	test_set_gray = (zeros((len(test_set[0]),1024)), test_set[1])

	for i in range(0,len(test_set[0])):
		print '\r', round((i + 1) * 100 / len(test_set[0]), 0), '%',
		for j in range(0,1024):
			test_set_gray[0][i][j] = 0.21 * test_set[0][i][j] + 0.71 * test_set[0][i][1024*1+j] + 0.07 * test_set[0][i][1024*2+j]
	print 'done!'

	print 'dumping test data into \'test_set_gray.pkl\'...'
	pickle(test_set_gray, 'data/test_set_gray.pkl')
