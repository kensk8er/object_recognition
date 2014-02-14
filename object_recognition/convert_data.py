#!/usr/local/bin/python

import var_dump

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
	
	print 'dumping training data into \'train_set.pkl\'...'
	train_set = (data_data, data_labels)
	pickle(train_set, 'data/train_set.pkl')

	# load validation data
	print 'loading original validation data...'
	data = unpickle(path + '/data_batch_5')

	print 'dumping validation data into \'valid_set.pkl\'...'
	valid_set = (data['data'] / 255., data['labels'])
	pickle(valid_set, 'data/valid_set.pkl')
	
	# load test data
	print 'loading original test data...'
	data = unpickle(path + '/test_batch')

	print 'dumping test data into \'test_set.pkl\'...'
	test_set = (data['data'] / 255., data['labels'])
	pickle(test_set, 'data/test_set.pkl')
