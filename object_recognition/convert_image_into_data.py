#!/usr/local/bin/python

from PIL import Image
from var_dump import var_dump
from unpickle import unpickle
from pickle import pickle
import numpy

# main
if __name__ == '__main__':
	# the number of test images
	image_num = 300000
	batch_size = 1000.
	batch_num = int(image_num / batch_size)

	color_data = [[0 for j in xrange(0,3072)] for i in xrange(0,int(batch_size))]
	
	print 'converting image data into pixel...'

	for i in xrange(0, batch_num):
		print str(i+1), '/', str(batch_num)

		for j in xrange(0, int(batch_size)):
			file_num = str(int(i * batch_size + (j + 1)))
			img = Image.open('test_image/' + file_num + '.png')
		
			rgbimg = img.convert("RGB")
			rgb = numpy.asarray(rgbimg.getdata(), dtype=numpy.float64)
			
			for k in xrange(0, len(rgb)):
				color_data[j][k] = rgb[k][0] / 255.
				color_data[j][k + 1024] = rgb[k][1] / 255.
				color_data[j][k + 2048] = rgb[k][2] / 255.
	
		color_data = numpy.asarray(color_data)
		file_num = str(i + 1)
		pickle(color_data, 'data/test_set_' + file_num + '.pkl')
	
	print 'done!'
	
