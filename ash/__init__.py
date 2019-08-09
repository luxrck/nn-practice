import itertools
from collections import OrderedDict
import numpy as np



# Activation Functions
def  softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    ex = np.exp(x)
    ez = np.exp(-x)
    return (ex - ez) / (ex + ez)


def d_sigmoid(sigmoid, z):
    return sigmoid(z) * (1 - sigmoid(z))
def d_tanh(tanh, z):
    return 1 - (tanh(z) * tanh(z))


def derivative(fn_name, *args):
    if fn_name == "sigmoid":
        return d_sigmoid(*args)
    elif fn_name == "tanh":
        return d_tanh(*args)


class Module(object):
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        self.__dict__[name] = value
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    def modules(self):
        return self._modules.values()
    def parameters(self, recurse=True):
        if recurse:
            for _,v in self._modules.items():
                for p in v.parameters():
                    yield p
        for p in self._parameters.values():
            yield p
    def named_parameters(self, recurse=True):
        pass



class Parameter(np.ndarray):
    pass



# Activation Function(linear): a = z = w * x + b
class Linear(Module):
    def __init__(self, i_size, j_size, bias=True):
        self._i_size = i_size
        self._j_size = j_size
        self._w = Parameter(np.random.randn(self._j_size,
                                            self._i_size))
        self._b = Parameter(np.random.randn(self._j_size, 1))
        # Temporary variables.
        self._x = None


    def forward(self, x):
        self._x = x
        return self._w.dot(x) + self._b


    def backward(self):
        return (self._x, 1)
