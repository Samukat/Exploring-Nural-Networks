import numpy as np

class sigmoid():
    name = "sigmoid"
    @staticmethod
    def __call__(z):
        return 1/(1+np.exp(-z))
    @staticmethod
    def derivative(z):
        return np.exp(z)/np.square((np.exp(z)+1))
    @staticmethod
    def inverse(z):
        return np.log(z/(1-z))
    
class RELU():
    name = "RELU"
    @staticmethod
    def __call__(z):
        return np.maximum(0,z)
    @staticmethod
    def derivative(z):
        return np.where(z>0,1,0)
    
class leaky_RELU():
    name = "leaky_RELU"
    @staticmethod
    def __call__(z):
        return np.where(z>0,z,z*0.01)
    @staticmethod
    def derivative(z):
        return np.where(z>0,1,0.01)
    
class mixed_RELU():
    name = "mixed_RELU"
    @staticmethod
    def __call__(z):
        return np.maximum(0,z)
    @staticmethod
    def derivative(z):
        return np.where(z>0,1,0.01)
    
class tanh():
    name = "tanh"
    @staticmethod
    def __call__(z):
        return np.tanh(z)
    @staticmethod
    def derivative(z):
        return 1 - np.square(np.tanh(z))

modules = {
    sigmoid.name:sigmoid,
    RELU.name:RELU,
    leaky_RELU.name:leaky_RELU,
    mixed_RELU.name:mixed_RELU,
    tanh.name:tanh
    }
