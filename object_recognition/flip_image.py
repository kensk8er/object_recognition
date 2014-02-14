#!/usr/local/bin/python

from PIL import Image
from var_dump import var_dump
from unpickle import unpickle
from pickle import pickle
import numpy

# main
if __name__ == '__main__':

	print 'loading the train set...'
	train_set = unpickle('data/train_set.pkl')
	image_num = len(train_set[0])
	train_set_orig = [[0 for j in xrange(32 * 32 * 3)] for i in xrange(image_num)]
	train_set_flip = [[0 for j in xrange(32 * 32 * 3)] for i in xrange(image_num)]
	train_set_y = train_set[1]
	#pickle(train_set_y, 'data/train_set_y.pkl')

	batch = 1
	print 'flipping the images'
	for i in xrange(image_num):
		print str(i+1), '/', str(len(train_set[0]))

		for j in xrange(32):

			for k in xrange(3):
				train_set_flip[i][k * 1024 + j * 32 : k * 1024 + (j + 1) * 32] = train_set[0][i][k * 1024 + j * 32 : k * 1024 + (j + 1) * 32][::-1]
				train_set_orig[i][k * 1024 + j * 32 : k * 1024 + (j + 1) * 32] = train_set[0][i][k * 1024 + j * 32 : k * 1024 + (j + 1) * 32]

		if (i + 1) % 500 == 0:
			index = (i + 1) / 500
			train_set_flip_x = train_set_orig[i + 1 - 500 : i + 1] + train_set_flip[i + 1 - 500 : i + 1]
			train_set_flip_x = numpy.asarray(train_set_flip_x, dtype=numpy.float64)
			train_set_flip_y = train_set_y[i + 1 - 500 : i + 1] + train_set_y[i + 1 - 500 : i + 1]

			print 'saving the flipped image'
			pickle(train_set_flip_x, 'data/train_set_flip_x_' + str(index) + '.pkl')
			pickle(train_set_flip_y, 'data/train_set_flip_y_' + str(index) + '.pkl')
			print 'done!'

	
