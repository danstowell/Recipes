
import numpy as np

import copy

import theano
import theano.tensor as T

import lasagne
from lasagne.layers.base import Layer

###############################################################################################################

def softabsT(x):
	"WARNING: this may blow up if given values of large magnitude. for float32, <80 preferred."
	return T.log(T.exp(x)+T.exp(-x))-T.log(2 * T.exp(0))

def qsoftabsT(x):
	"a 'soft-abs' formed by gluing together quadratic and linear regions"
	x = abs(x)
	return T.maximum((x*x)*(x<0.5), (x-0.25))

"""""
def softplusT(x):
	return T.log(T.exp(x) + 1)

def unsoftplusT(x):
	return T.log(T.exp(x) - 1)

def softplusN(x):
	return np.log(np.exp(x) + 1)

def unsoftplusN(x):
	return np.log(np.exp(x) - 1)
"""

###############################################################################################################

class Conv2DLayerPlus(lasagne.layers.Conv2DLayer):
	"Conv2DLayer, but kitted out with getters & setters to enable subclasses to do weird things"
	def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
			pad=0, untie_biases=False,
			W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
			nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
			convolution=T.nnet.conv2d, **kwargs):
		super(Conv2DLayerPlus, self).__init__(incoming, num_filters, filter_size,
						stride, pad, untie_biases, W, b,
						nonlinearity, flip_filters, convolution,
						**kwargs)

		self.normalise_filters = self.define_filter_normalisation_function(self.W)
		self.__hardproject_filters_functions = {}
		self.get_filters = theano.function([], self.W.dimshuffle(1, 2, 3, 0)[0])   # this shuffle drops the unused second dim. not sure why needed. check shufflogic.
		self.get_filters_noshuffle = theano.function([], self.W)

	def define_filter_normalisation_function(self, filtervar):
		"Returns a theano function that when invoked will normalise the filters"
		normval = 1. / (1e-7 + T.sqrt(T.sum(filtervar * filtervar, axis=(0,1,3), keepdims=True))) # theano expression for a multiplicative update to the filters
		return theano.function([], normval, updates=[(filtervar, filtervar * normval)]) # fixes L2 norm of each filter to unity

	def hardproject_filters(self, projmethod):
		if projmethod not in self.__hardproject_filters_functions: # lazy init to avoid order-of-operations problem in constructor
			if projmethod=='abs':
				self.__hardproject_filters_functions['abs'] = self.define_hardproject_abs_function()
			elif projmethod=='relu':
				self.__hardproject_filters_functions['relu'] = self.define_hardproject_relu_function()
			else:
				raise ValueError("Unknown hard-projection name: '%s'" % projmethod)
		return self.__hardproject_filters_functions[projmethod]()

	def define_hardproject_abs_function(self):
		"Returns a theano function that when invoked will hard-project the filters to enforce nonnegativity"
		return theano.function([], [], updates=[(self.W, abs(self.W))])

	def define_hardproject_relu_function(self):
		"Returns a theano function that when invoked will hard-project the filters to enforce nonnegativity"
		return theano.function([], [], updates=[(self.W, T.maximum(0, self.W))])

	def set_filter_values(self, vals):
		self.W.set_value(vals)

	def get_filter_values(self):
		return self.get_filters()

	def get_filter_values_noshuffle(self):
		return self.get_filters_noshuffle()

	def convolve(self, input, **kwargs):
		"Should be EXACTLY the same as Conv2DLayer:convolve() but calls out to self.get_thing_to_convolve()"
		border_mode = 'half' if self.pad == 'same' else self.pad
		conved = self.convolution(input, self.get_thing_to_convolve(),
					self.input_shape, self.get_W_shape(),
					subsample=self.stride,
					border_mode=border_mode,
					filter_flip=self.flip_filters)
		return conved

	def get_thing_to_convolve(self):
		"subclasses can override this if they need the convolutive filters to be something other than purely the W variable"
		return self.W


class Conv2DLayerNonNeg(Conv2DLayerPlus):
	"""A simple tweak - exactly Conv2DLayer but with nonnegative convolutions.
	The nonnegativity is enforced via the "nonlinearityW" operation, a static nonlinearity. User-settable, but the intention is for it to be a nonneg outcome.
	"""

	def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
			pad=0, untie_biases=False,
			W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
			nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
			convolution=T.nnet.conv2d, nonlinearityW=abs, **kwargs):
		super(Conv2DLayerNonNeg, self).__init__(incoming, num_filters, filter_size,
						stride, pad, untie_biases, W, b,
						nonlinearity, flip_filters, convolution,
						**kwargs)
		self.nonlinearityW = nonlinearityW

	def get_thing_to_convolve(self):
		return self.nonlinearityW(self.W)

