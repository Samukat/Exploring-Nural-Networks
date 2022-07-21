###original attemps at a network though a Node class
import numpy as np


class Node:
    def __init__(self, inputs=None):
        self.bias = 0
        self.inputs = None
        self.weights = None

        if inputs != None:
            self.inputs = inputs
            self.weights = np.ones((len(inputs),))

    def get_output(self, inputs=None):
        #print(self.inputs)
        #inputnodevar = [a.get_output() for a in self.inputs]
        #print(inputnodevar)

        if type(self.inputs[0]) == Node:
            inputnodevar = [a.get_output() for a in self.inputs]
        else:
            inputnodevar = self.inputs

        
        if inputs != None:
            inputnodevar = inputs

        mult = np.multiply(inputnodevar, self.weights) 
        out = np.sum(mult) + self.bias

        print(out)
        return out

        

start1 = Node([1])
start2 = Node([2])
start3 = Node([3])

mid1 = Node([start1,start2,start3])
mid2 = Node([start1,start2,start3])
mid3 = Node([start1,start2,start3])

out1 = Node([mid1,mid2,mid3])
print(out1.get_output())




