import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

random.seed(1)
from tensorflow.keras.datasets import mnist

class Network():
    def __init__(self, internal_architecture):
        self.learning_rate = 0.1
        self.internal_architecture = internal_architecture
        self.total_layers = len(internal_architecture)
        
        inputNodes = internal_architecture[0]
        outputNodes = internal_architecture[-1]

        network_values = {}
        
        for index, layer in enumerate(internal_architecture[1:]):
            if index == 0:
                layer_input_size = inputNodes["inputNodes"]
            else:
                layer_input_size = internal_architecture[index]["hiddenNodes"]

            if index == self.total_layers-2:
                nodes = layer["outputLayers"]
            else:
                nodes = layer["hiddenNodes"]

            network_values['w' + str(index)] = np.random.randn(nodes, layer_input_size) * 0.1
            network_values['b' + str(index)] = np.random.randn(nodes) * 0.1
            network_values['w' + str(index)].dtype = np.float64
            network_values['w' + str(index)].dtype = np.float64

 
        self.network_values = network_values
        #print(network_values)
        #print(internal_architecture)


    ##propergation
    ##
    def single_layer_forward_propagate(self, inputs, weights, baises, activation="sigmoid"):
        output = np.dot(weights, inputs) + baises
        return Network.activation(output)
    
    def forward_propagate(self, inputs):
        memory = [inputs] #[inputs] <-- optional, but changes index
        current_values = inputs

        for index, layer in enumerate(internal_architecture[1:]):
            current_values = self.single_layer_forward_propagate(
                current_values,
                self.network_values['w' + str(index)],
                self.network_values['b' + str(index)]
            )
            #print(current_values)
            memory.append(current_values)
        self.last_memory = memory #use for back prop (optimisation so that we dont have to reprop), may get ram instensive....
        return current_values

        
    ##training
    ##
    def BackPropergate(self, inputs, expected):
        errors = self.divloss(network.forward_propagate(inputs),expected)
        #print(self.last_memory)
        
        #derivitiveMemory = []
        #for i, error in enumerate(errors): ##each output node
         #   z = np.dot(self.network_values["w"+str(self.total_layers-2)][i], self.last_memory[-2]) + self.network_values["b"+str(self.total_layers-2)][i] #-2 as -1 is last, last is output (for memory), -2 on network values as total includes input and oputput layers
          #  derivitiveMemory.append(error * Network.divActivation(z,"sigmoid")) #dc/dz(L)  =  dc/da(L) * da(L)/dZ(L)
                #derivitive momeory from L+1 (not including final multiplication)
        
        #z = np.dot(self.network_values["w"+str(self.total_layers-2)], self.last_memory[-2]) + self.network_values["b"+str(self.total_layers-2)] #-2 as -1 is last, last is output (for memory), -2 on network values as total includes input and oputput layers
        partial_derivitive_memory = errors 

        #print("\n")
        #print(partial_derivitive_memory)
        #print("\n")
        
        for layer in range(self.total_layers-2, -1, -1):
            #print("weights at l: ",self.network_values["w"+str(layer)])
            #print("partial derivities: ",partial_derivitive_memory)
            #weight_corrections = np.dot(np.diag(partial_derivitive_memory), np.absolute(self.network_values["w"+str(layer)])) ##why tf is this absolute

            z = np.dot(self.network_values["w"+str(layer)], self.last_memory[layer]) + self.network_values["b"+str(layer)] #-2 as -1 is last, last is output (for memory), -2 on network values as total includes input and oputput layers
            partial_derivitive_memory = partial_derivitive_memory* Network.divActivation(z,"sigmoid")
            #print("z2", z)

            #print(layer)
            #print("partial", partial_derivitive_memory[np.newaxis].T, "and mem", np.array(self.last_memory[layer])[np.newaxis])
            weight_corrections = np.matmul(partial_derivitive_memory[np.newaxis].T, np.array(self.last_memory[layer])[np.newaxis])
            #print("prop memory: ", self.last_memory[-layer])
            #print("corrections:", weight_corrections)
            #print("pdir", partial_derivitive_memory)
            #print("abises", self.network_values["b"+str(layer)])
            #print(self.network_values["w"+str(layer)])
            bias_corrections = partial_derivitive_memory
            self.network_values["b"+str(layer)] -= self.learning_rate*bias_corrections

            #print(np.shape(partial_derivitive_memory[np.newaxis].T), np.shape(self.network_values["w"+str(layer)]))
            #print("aa", np.dot(np.diag(partial_derivitive_memory), self.network_values["w"+str(layer)]))
            #print("bb", np.dot(partial_derivitive_memory, self.network_values["w"+str(layer)]))
            
            partial_derivitive_memory = np.dot(partial_derivitive_memory, self.network_values["w"+str(layer)]) #i removed the diag cuz add them together?

            
            #print(self.network_values["w"+str(layer)])
            self.network_values["w"+str(layer)] -= self.learning_rate*weight_corrections 


            
            

            #partial_derivitive_memory = np.dot(np.diag(partial_derivitive_memory),)



            
            #self.network_values["w"+str(layer)] - 1
            


            
            #for i, partial_derivitive in enumerate(derivitiveMemory):
                #print("w"+str(layer),self.network_values["w"+str(layer)])
                
           
                



                
            #print("b"+str(layer),self.network_values["b"+str(layer)])

                

            
                
            

    ##Loss and activations
    ##
    def loss(self, predicted, expected): #expected as array... e.g. 5 = [0,0,0,0,1,0,0,0,0,0]
        #total loss in matrix
        #takes in len()
        if len(expected) != len(predicted): ##checking exp is the same lenth as the output layers 
            raise AttributeError("len exp != len output")
        
        outputs = []
        for pred, exp in zip(predicted,expected):
            outputs.append((exp - pred)**2)

        return outputs
    
    def divloss(self, predicted, expected): #expected as array... e.g. 5 = [0,0,0,0,1,0,0,0,0,0] 
        #total loss in matrix
        #takes in len()
        if len(expected) != len(predicted): ##checking exp is the same lenth as the output layers 
            raise AttributeError("len exp != len output")
        
        outputs = []
        for pred, exp in zip(predicted,expected):
            outputs.append(2*(pred - exp))
            
        return outputs
    def activation(inp):
        inp.dtype = np.longdouble
        #inp = np.clip(inp, a_min=-150, a_max=150)
        return 1/(1+np.exp(-inp))

        
        #if function == "sigmoid"  

    def divActivation(inp, function: str): ##da/dz z = imp
        inp.dtype = np.longdouble
        #inp = np.clip(inp, a_min=-150, a_max=150)
        #print(max(inp))
        if function == "sigmoid":
            return np.exp(-(inp))/(np.exp(-(inp))+1)**2

        elif function == "relu":
            return 1

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axarr = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axarr.flat):
        #ax.imshow(images[i], cmap='binary')
        ax.imshow(images[i])
        
        if cls_pred is not None:
            ax.set_xlabel("True: {}, Pred: {}".format(cls_true[i], cls_pred[i]))
        else:
            ax.set_xlabel("True: {}".format(cls_true[i]))

        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def convertToExpected(numIn):
    output = [0]*10
    output[numIn-1] = 1
    return output



if __name__ == "__main__":
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))
    #plot_images(train_X[0:9], train_y[0:9])

    internal_architecture = [
        {"inputNodes": 784},
        #{"hiddenNodes": 16, "activation": "sigmoid"},
        #{"hiddenNodes": 64, "activation": "sigmoid"},
        {"hiddenNodes": 32, "activation": "sigmoid"},
        {"outputLayers": 10, "activation": "sigmoid"}
    ]
    network = Network(internal_architecture)
    inputs = train_X[0].flatten()/255
    #print(max(inputs))
    expected = convertToExpected(train_y[0])
    output = network.forward_propagate(inputs)
    #print(output.argmax(), train_y[0])
    #print("loss", sum(network.loss(output,expected)))
    network.BackPropergate(inputs,expected)

   
    
    
    for i in range(60000):
        inputs = train_X[i].flatten()/255
        expected = convertToExpected(train_y[i])
        network.BackPropergate(inputs,expected)

    output = network.forward_propagate(train_X[0].flatten()/255)
    print(output.argmax(), train_y[0])
    print("loss", sum(network.loss(output,expected)))

    plot_images(test_X[0:9], test_y[0:9])
    correct = 0
    pp = [1,2,3,4,5,6,7,8,9,0]
    testxincorrect = []
    testyincorrect = []
    pred = []
    for i in range(10000):
        inputs = test_X[i].flatten()/255
        expected = convertToExpected(test_y[i])
        output = network.forward_propagate(inputs)
        #print(output)

        
        if pp[output.argmax()] == test_y[i]:
            correct += 1
        else:
            testxincorrect.append(test_X[i])
            testyincorrect.append(test_y[i])
            pred.append(pp[output.argmax()])
    print(correct, correct/10000)
    plot_images(testxincorrect[0:9], testyincorrect[0:9], pred)
































if __name__ == "__maain__":
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))
    #plot_images(train_X[0:9], train_y[0:9])

    internal_architecture = [
        {"inputNodes": 2},
        {"hiddenNodes": 3, "activation": "sigmoid"},
        {"hiddenNodes": 3, "activation": "sigmoid"},
        {"outputLayers": 1, "activation": "sigmoid"}
    ]
    network = Network(internal_architecture)


    for i in range(100000):
        a = random.random()
        b = random.random()
        inputs = [a,b]
        if (a>b):
            expected = 1
        else:
            expected = 0
        network.BackPropergate(inputs,[expected])


    while True:
        a = float(input())
        b = float(input())
        inputs = [a,b]
        output = network.forward_propagate(inputs)
        print(output)




















    
    #
    #output = loss(networktrain_X[0].flatten()), train_Y[0])
    #print(train_y[0])
    #output = network.loss(network.forward_propagate(train_X[0].flatten()), convertToExpected(train_y[0]))

    #inputs = [3,2,1,4,5]
    #expected = [0,1,0]

    
    #print("loss", sum(network.loss(network.forward_propagate(inputs),expected)))

    #print(network.forward_propagate(inputs),expected)
    #print()
    #for i in range(1):
    #    network.BackPropergate(inputs,expected)
    #print()
    #print(network.forward_propagate(inputs),expected)
    #print("loss", sum(network.loss(network.forward_propagate(inputs),expected)))



    #print(network.network_values)
    #print(output.argmax())
    


