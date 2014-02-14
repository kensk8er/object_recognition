#!/usr/local/bin/python

import var_dump
import numpy

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
	## load the training data
	print 'loading training data...'
	train_set = unpickle('data/train_set.pkl')
	train_set_y = train_set[1]
	train_set_color = [[[0 for z in xrange(1024)] for y in xrange(3)] for x in xrange(len(train_set[0]))]

	print 'converting training data into color format...'
	for i in xrange(len(train_set[0])):
		print str(i+1), '/', str(len(train_set[0]))
		data_color = [[0 for y in xrange(1024)] for x in xrange(3)]

		for j in xrange(3):
			data_color[j] = train_set[0][i][j*1024:(j+1)*1024]

		train_set_color[i] = data_color

	print 'done!'

	train_set_color = numpy.asarray(train_set_color)
	train_set_color = (train_set_color, train_set_y)

	print 'saving the color format...'

	pickle(train_set_color, 'data/train_set_color.pkl')

	print 'done!'



	## load the validation data
	print 'loading validation data...'
	valid_set = unpickle('data/valid_set.pkl')
	valid_set_y = valid_set[1]
	valid_set_color = [[[0 for z in xrange(1024)] for y in xrange(3)] for x in xrange(len(valid_set[0]))]

	print 'converting validing data into color format...'
	for i in xrange(len(valid_set[0])):
		print str(i+1), '/', str(len(valid_set[0]))
		data_color = [[0 for y in xrange(1024)] for x in xrange(3)]

		for j in xrange(3):
			data_color[j] = valid_set[0][i][j*1024:(j+1)*1024]

		valid_set_color[i] = data_color

	print 'done!'

	valid_set_color = numpy.asarray(valid_set_color)
	valid_set_color = (valid_set_color, valid_set_y)

	print 'saving the color format...'

	pickle(valid_set_color, 'data/valid_set_color.pkl')

	print 'done!'





	## load the testing data
	print 'loading testing data...'
	test_set = unpickle('data/test_set.pkl')
	test_set_y = test_set[1]
	test_set_color = [[[0 for z in xrange(1024)] for y in xrange(3)] for x in xrange(len(test_set[0]))]

	print 'converting testing data into color format...'
	for i in xrange(len(test_set[0])):
		print str(i+1), '/', str(len(test_set[0]))
		data_color = [[0 for y in xrange(1024)] for x in xrange(3)]

		for j in xrange(3):
			data_color[j] = test_set[0][i][j*1024:(j+1)*1024]

		test_set_color[i] = data_color

	print 'done!'

	test_set_color = numpy.asarray(test_set_color)
	test_set_color = (test_set_color, test_set_y)

	print 'saving the color format...'

	pickle(test_set_color, 'data/test_set_color.pkl')

	print 'done!'
