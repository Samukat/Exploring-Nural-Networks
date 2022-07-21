import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import json
import sys

from tensorflow.keras.datasets import mnist



class NetworkFunctions(): ##convert to module(as opposed to class) later
    class sigmoid():
        @staticmethod
        def __call__(z):
            return 1/(1+np.exp(-z))
        @staticmethod
        def derivative(z):
            return np.exp(z)/np.square((np.exp(z)+1))
        @staticmethod
        def inverse(z):
            return np.log(z/(1-z))
        
        def toJSON(self):
            print(self)
            return json.dumps(self, default=lambda o: o.__dict__, 
                sort_keys=True, indent=4)
        
    class RELU():
        @staticmethod
        def __call__(z):
            return np.maximum(0,z)
        @staticmethod
        def derivative(z):
            return np.where(z>0,1,0)
        
    class leaky_RELU():
        @staticmethod
        def __call__(z):
            return np.where(z>0,z,z*0.01)
        @staticmethod
        def derivative(z):
            return np.where(z>0,1,0.01)
        
    class mixed_RELU():
        @staticmethod
        def __call__(z):
            return np.maximum(0,z)
        @staticmethod
        def derivative(z):
            return np.where(z>0,1,0.01)
        
    class tanh():
        @staticmethod
        def __call__(z):
            return np.tanh(z)
        @staticmethod
        def derivative(z):
            return 1 - np.square(np.tanh(z))
        

class Network():
    def __init__(self, internal_architecture, learning_rate=0.01):
        self.learning_rate = learning_rate
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

        self.network_values = network_values



    ##propergation
    ##
    def single_layer_forward_propagate(self, inputs, weights, baises, activation):
        inputs = np.atleast_2d(inputs).T
        output = np.dot(weights, inputs)
        output += baises[np.newaxis].T
        return activation(output.T)
    
    def forward_propagate(self, inputs):
        memory = [inputs] #[inputs] <-- optional, but changes index
        current_values = inputs

        for index, layer in enumerate(internal_architecture[1:]):
            current_values = self.single_layer_forward_propagate(
                current_values,
                self.network_values['w' + str(index)],
                self.network_values['b' + str(index)],
                layer["activation"]
            )
            memory.append(current_values)
        self.last_memory = memory #use for back prop (optimisation so that we dont have to reprop), may get ram instensive....
        return current_values


    def single_layer_inverse_propagate(self,inputs,weights,baises,activation):
        inputs = np.atleast_2d(inputs).T
        output = activation.inverse(z)
        output -= baises
        
    def inverse_propagate(self, inputs):
        reverse_values = self

        
    ##training
    ##
    def back_propergate(self, inputs, expected):
        errors = self.divloss(self.forward_propagate(inputs),expected)
        #print(errors,self.forward_propagate(inputs),expected)
        partial_derivitive_memory = errors
        
        for layer in range(self.total_layers-2, -1, -1):
            z = np.dot(self.network_values["w"+str(layer)], self.last_memory[layer].T)+ self.network_values["b"+str(layer)][np.newaxis].T
            a = self.internal_architecture[layer+1]["activation"].derivative(z)

            #print(partial_derivitive_memory)
            partial_derivitive_memory = (np.array(partial_derivitive_memory).T*a).T
            mem = np.array(self.last_memory[layer])
            #print(partial_derivitive_memory.T.shape,mem.shape)
            
            weight_corrections = np.matmul(partial_derivitive_memory.T,mem)
            bias_corrections = np.sum(partial_derivitive_memory,axis=0)
            #print(bias_corrections)

            self.network_values["b"+str(layer)] -= self.learning_rate*bias_corrections
            partial_derivitive_memory = np.dot(partial_derivitive_memory, self.network_values["w"+str(layer)])
            
            self.network_values["w"+str(layer)] -= self.learning_rate*weight_corrections
            #print("pass")

    def train(self, trainX, TrainY, batch_size=1):
        data = self.generate_batches(trainX, TrainY, batch_size)
        for x, y in data:
            self.back_propergate(x,y)
            
    def test(self, testX, testY):
        correct = 0
        testxincorrect = []
        testyincorrect = []
        pred = []
        data = self.generate_batches(testX, testY, 1)
        for x, y in data:
            output = network.forward_propagate(x)
            if np.argmax(output) == np.argmax(y):
                correct += 1
            else:
                testxincorrect.append(x.reshape(28,28))
                testyincorrect.append(np.argmax(y))
                pred.append(np.argmax(output))
                
        return (correct, testxincorrect, testyincorrect, pred)
            
        
    @staticmethod
    def generate_batches(trainX, trainY, batch_size):
        if len(trainX) != len(trainY):
            raise AttributeError("X and Y not same size")
        if (len(trainX)%batch_size != 0):
            print("warning: values dropped as total not divisable by batch_size")
        if (type(trainX) == list):
            trainX = np.array(trainX) ##add not if ndarray
        if (type(trainY) == list):
            trainY = np.array(trainY)
            
        batches = int(len(trainY)/batch_size)
        trainXBatches = np.zeros((batches, batch_size, trainX.shape[1]))
        trainYBatches = np.zeros((batches, batch_size, trainY.shape[1]))
        
        for i in range(batches):
            trainXBatches[i] = trainX[i*batch_size:(i+1)*batch_size]
            trainYBatches[i] = trainY[i*batch_size:(i+1)*batch_size]
            
        return zip(trainXBatches,trainYBatches)
        
        

    ##Loss and activations
    ##
    def loss(self, predicted, expected): #expected as array... e.g. 5 = [0,0,0,0,1,0,0,0,0,0]
        expected = np.atleast_2d(expected)
        predicted = np.atleast_2d(predicted)
        if expected.shape != predicted.shape: ##checking exp is the same lenth as the output layers 
            raise AttributeError("expected: " + str(expected.shape) + " is not predicted: " + str(predicted.shape))
        
        outputs = []
        for pred, exp in zip(predicted,expected):
            outputs.append((exp - pred)**2)

        return outputs
    
    def divloss(self, predicted, expected): #expected as array... e.g. 5 = [0,0,0,0,1,0,0,0,0,0] 
        expected = np.atleast_2d(expected)
        predicted = np.atleast_2d(predicted)
        if expected.shape != predicted.shape: ##checking exp is the same lenth as the output layers 
            raise AttributeError("expected: " + str(expected.shape) + " is not predicted: " + str(predicted.shape))
        
        outputs = []
        for pred, exp in zip(predicted,expected):
            outputs.append(2*(pred - exp))
        
        return outputs


    ###Exports and Imports
    def export_network(self, network_name="network_save"):
        
        pass



        
##        data = {"internal_architecture":self.internal_architecture, "learning_rate":self.learning_rate, "network_values":self.network_values}
##        print(data)
##        datadump = json.dumps(data)
##        with open(network_name, 'w+') as f:
##            json.dump(data, f)
##        return datadump
    
    def import_netwrok(self,json_network):
        pass
    


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





test = False

if __name__ == "__main__" and test:

    internal_architecture = [
        {"inputNodes": 1},
        {"hiddenNodes": 2, "activation": NetworkFunctions.sigmoid()},
        {"hiddenNodes": 2, "activation": NetworkFunctions.sigmoid()},
        {"outputLayers": 1, "activation": NetworkFunctions.leaky_RELU()}
    ]
    network = Network(internal_architecture)
    inputs = [[0],[0]]
    expected = [[1],[0]]

    print(sum(network.loss(network.forward_propagate(inputs), expected)))
    print(network.loss(network.forward_propagate(inputs), expected))
    print(network.network_values)
    for a in range(60000):
        network.train(inputs,expected,16)
    print(network.network_values)
    print(sum(network.loss(network.forward_propagate(inputs), expected)))
    print(network.forward_propagate([1]))
    

     


if __name__ == "__main__" and not test:
    random.seed(1)
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))
    #plot_images(train_X[0:9], train_y[0:9])

    ##Setup
    internal_architecture = [
        {"inputNodes": 784},
        {"hiddenNodes": 128, "activation": NetworkFunctions.sigmoid()},
        {"hiddenNodes": 64, "activation": NetworkFunctions.sigmoid()},
        {"outputLayers": 10, "activation": NetworkFunctions.sigmoid()}
    ]
    network = Network(internal_architecture, learning_rate=0.15)

    #Train
    amount = 60000
    train_X = train_X[0:amount]
    train_y = train_y[0:amount]
    flat_x = train_X.flatten().reshape(amount,784)/255
    flat_y = np.zeros((amount,10), dtype=int)
    for i, value in enumerate(train_y):
        flat_y[i][value] = 1
    network.train(flat_x,flat_y,8)

    #Test
    amount = 10000
    test_X = test_X[0:amount]
    test_y = test_y[0:amount]
    flat_x = test_X.flatten().reshape(amount,784)/255
    flat_y = np.zeros((amount,10), dtype=int)
    for i, value in enumerate(test_y):
        flat_y[i][value] = 1

    test_results = network.test(flat_x,flat_y)
    print(test_results[0])
    plot_images(test_results[1][0:9], test_results[2][0:9], test_results[3])










#####Notes and findings:
    #when the batch size without change of learning rate is increased the amount of change for a single item tended to over correcct, i.e large predictions of 4s and 8s. 
