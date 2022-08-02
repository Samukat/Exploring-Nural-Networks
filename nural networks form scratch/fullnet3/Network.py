import numpy as np
import network_functions as nf
import random
import json
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#from numba import jit, cuda

class Network():    
    def __init__(self, internal_architecture, learning_rate=0.01, momentum=0, decay=0.01):
        self.internal_architecture = internal_architecture
        for layer in self.internal_architecture:
            if "activation" in layer:
                layer['activation_class'] = nf.modules[layer['activation']]()
                

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.smoothing_rate = 10e-8
 
        self.total_layers = len(internal_architecture)
        inputNodes = internal_architecture[0]
        outputNodes = internal_architecture[-1]

        network_values = {}
        velocity = {} #for momentum optimisation
        gradient_sums = {}
        
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


            ##creating 0ed dictionaries for velocity (momentum optimisation), gradient_sums(adagrad learning rate)
            velocity['w' + str(index)] = np.zeros((nodes, layer_input_size))
            velocity['b' + str(index)] = np.zeros((nodes))
            gradient_sums['w' + str(index)] = np.zeros((nodes, layer_input_size))
            gradient_sums['b' + str(index)] = np.zeros((nodes))

        self.network_values = network_values
        self.velocity = velocity
        self.gradient_sums = gradient_sums



    ##propergation
    ##
    def single_layer_forward_propagate(self, inputs, weights, biases, activation):
        inputs = np.atleast_2d(inputs).T
        output = np.dot(weights, inputs)
        output += biases[np.newaxis].T
        return activation(output.T)
    
    def forward_propagate(self, inputs):
        memory = [inputs] #[inputs] <-- optional, but changes index
        current_values = inputs

        for index, layer in enumerate(self.internal_architecture[1:]):
            current_values = self.single_layer_forward_propagate(
                current_values,
                self.network_values['w' + str(index)],
                self.network_values['b' + str(index)],
                layer["activation_class"]
            )
            memory.append(current_values)
        self.last_memory = memory #use for back prop (optimisation so that we dont have to reprop), may get ram instensive....
        return current_values


    def single_layer_inverse_propagate(self,inputs,weights,biases,activation):
        inputs = np.atleast_2d(inputs).T
        output = activation.inverse(z)
        output -= biases
        
    def inverse_propagate(self, inputs):
        reverse_values = self

        
    ##optimisation
    ##
    def back_propergate(self, inputs, expected): ##gradient Decenent
        errors = self.divloss(self.forward_propagate(inputs),expected)
        loss = np.sum(self.loss(self.forward_propagate(inputs),expected))
        partial_derivitive_memory = errors
        
        for layer in range(self.total_layers-2, -1, -1):
            z = np.dot(self.network_values["w"+str(layer)], self.last_memory[layer].T)+ self.network_values["b"+str(layer)][np.newaxis].T
            a = self.internal_architecture[layer+1]["activation_class"].derivative(z)

            partial_derivitive_memory = (np.array(partial_derivitive_memory).T*a).T
            mem = np.array(self.last_memory[layer])
            
            weight_corrections = np.matmul(partial_derivitive_memory.T,mem)
            bias_corrections = np.sum(partial_derivitive_memory,axis=0)

            self.network_values["b"+str(layer)] -= self.learning_rate*bias_corrections
            partial_derivitive_memory = np.dot(partial_derivitive_memory, self.network_values["w"+str(layer)])
            
            self.network_values["w"+str(layer)] -= self.learning_rate*weight_corrections
        return loss

    def SGD(self, inputs, expected): #with momentum 
        errors = self.divloss(self.forward_propagate(inputs),expected)
        loss = np.sum(self.loss(self.forward_propagate(inputs),expected))
        partial_derivitive_memory = errors

        for layer in range(self.total_layers-2, -1, -1):
            z = np.dot(self.network_values["w"+str(layer)], self.last_memory[layer].T)+ self.network_values["b"+str(layer)][np.newaxis].T
            a = self.internal_architecture[layer+1]["activation_class"].derivative(z)

            partial_derivitive_memory = (np.array(partial_derivitive_memory).T*a).T
            mem = np.array(self.last_memory[layer])

            ##momentum - changes(L) = Lrate*grad(L) + B*changes(L-1)
            weight_corrections = self.momentum*self.velocity["w"+str(layer)] + self.learning_rate*np.matmul(partial_derivitive_memory.T,mem)
            bias_corrections = self.momentum*self.velocity["b"+str(layer)] + self.learning_rate*np.sum(partial_derivitive_memory,axis=0)

            ##momentum with exponential average   Wc = B*v + (1-B)*.........
            #weight_corrections = self.momentum*self.velocity["w"+str(layer)] + (1-self.momentum)*self.learning_rate*np.matmul(partial_derivitive_memory.T,mem) #bias correction for exponential average not implemetned
            #bias_corrections = self.momentum*self.velocity["b"+str(layer)] + (1-self.momentum)*self.learning_rate*np.sum(partial_derivitive_memory,axis=0) #bias correction for exponential average not implemetned
            
            self.network_values["b"+str(layer)] -= bias_corrections
            self.velocity["b"+str(layer)] = bias_corrections
            
            partial_derivitive_memory = np.dot(partial_derivitive_memory, self.network_values["w"+str(layer)])
                
                
            self.network_values["w"+str(layer)] -= weight_corrections
            self.velocity["w"+str(layer)] = weight_corrections
        return loss


    def adagrad(self, inputs, expected):
        #https://datascience.stackexchange.com/questions/82116/why-are-we-taking-the-square-root-of-the-gradient-in-adagrad
        #https://ruder.io/optimizing-gradient-descent/index.html
        #https://aclanthology.org/D17-1046.pdf
        #https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c
        errors = self.divloss(self.forward_propagate(inputs),expected)
        loss = np.sum(self.loss(self.forward_propagate(inputs),expected))
        partial_derivitive_memory = errors
        for layer in range(self.total_layers-2, -1, -1):
            z = np.dot(self.network_values["w"+str(layer)], self.last_memory[layer].T)+ self.network_values["b"+str(layer)][np.newaxis].T
            a = self.internal_architecture[layer+1]["activation_class"].derivative(z)

            partial_derivitive_memory = (np.array(partial_derivitive_memory).T*a).T
            mem = np.array(self.last_memory[layer])

            #gradients
            weight_gradient = np.matmul(partial_derivitive_memory.T,mem)
            bias_gradients = np.sum(partial_derivitive_memory,axis=0)

            
            #update gradient square sums
            self.gradient_sums["w"+str(layer)] += np.square(weight_gradient)
            self.gradient_sums["b"+str(layer)] += np.square(bias_gradients)

            #update learning rates
            adapted_learning_rate_weights = (self.learning_rate)/np.sqrt(self.gradient_sums["w"+str(layer)]+self.smoothing_rate)
            adapted_learning_rate_biases  = (self.learning_rate)/np.sqrt(self.gradient_sums["b"+str(layer)]+self.smoothing_rate)
            #print(adapted_learning_rate_weights[0][0],adapted_learning_rate_weights[1][1])
            
            #print(adapted_learning_rate_weights.shape)
            #sprint(weight_gradient.shape)
            
            weight_corrections = self.momentum*self.velocity["w"+str(layer)] + adapted_learning_rate_weights*weight_gradient
            bias_corrections = self.momentum*self.velocity["b"+str(layer)] + adapted_learning_rate_biases*bias_gradients

            
            
            #update velocities
            self.network_values["b"+str(layer)] -= bias_corrections
            self.velocity["b"+str(layer)] = bias_corrections
        
            partial_derivitive_memory = np.dot(partial_derivitive_memory, self.network_values["w"+str(layer)])
            
            
            self.network_values["w"+str(layer)] -= weight_corrections
            self.velocity["w"+str(layer)] = weight_corrections

        return loss


    def adadelta(self, inputs, expected):
        #use gradient_sums as decayed sum thingo

        partial_derivitive_memory = self.divloss(self.forward_propagate(inputs),expected)
        loss = np.sum(self.loss(self.forward_propagate(inputs),expected))
        
        for layer in range(self.total_layers-2, -1, -1):
            z = np.dot(self.network_values["w"+str(layer)], self.last_memory[layer].T)+ self.network_values["b"+str(layer)][np.newaxis].T
            a = self.internal_architecture[layer+1]["activation_class"].derivative(z)

            partial_derivitive_memory = (np.array(partial_derivitive_memory).T*a).T
            mem = np.array(self.last_memory[layer])

            #gradients 
            weight_gradients = np.matmul(partial_derivitive_memory.T,mem)
            bias_gradients = np.sum(partial_derivitive_memory,axis=0)

            
            #update gradient decayed sums (exponentail average)
            self.gradient_sums["w"+str(layer)] = (self.decay)*self.gradient_sums["w"+str(layer)] + (1-self.decay)*np.square(weight_gradients) 
            self.gradient_sums["b"+str(layer)] = (self.decay)*self.gradient_sums["b"+str(layer)] + (1-self.decay)*np.square(bias_gradients) 

            #update learning rates
            adapted_learning_rate_weights = (self.learning_rate)/np.sqrt(self.gradient_sums["w"+str(layer)]+self.smoothing_rate)
            adapted_learning_rate_biases  = (self.learning_rate)/np.sqrt(self.gradient_sums["b"+str(layer)]+self.smoothing_rate)
            
            weight_corrections = self.momentum*self.velocity["w"+str(layer)] + (1-self.momentum)*adapted_learning_rate_weights*weight_gradients
            bias_corrections = self.momentum*self.velocity["b"+str(layer)] + (1-self.momentum)*adapted_learning_rate_biases*bias_gradients

            
            #update velocities
            self.network_values["b"+str(layer)] -= bias_corrections
            self.velocity["b"+str(layer)] = bias_corrections
        
            partial_derivitive_memory = np.dot(partial_derivitive_memory, self.network_values["w"+str(layer)])
            
            
            self.network_values["w"+str(layer)] -= weight_corrections
            self.velocity["w"+str(layer)] = weight_corrections

        return loss

    def ADAM(self, inputs, expected):         #https://arxiv.org/pdf/1412.6980.pdf
        partial_derivitive_memory = self.divloss(self.forward_propagate(inputs),expected)
        loss = np.sum(self.loss(self.forward_propagate(inputs),expected))
        
        for layer in range(self.total_layers-2, -1, -1):
            ##where f(z) = a(z), z = a(z)(l-1) + b
            z = np.dot(self.network_values["w"+str(layer)], self.last_memory[layer].T)+ self.network_values["b"+str(layer)][np.newaxis].T
            a = self.internal_architecture[layer+1]["activation_class"].derivative(z)

            partial_derivitive_memory = (np.array(partial_derivitive_memory).T*a).T
            mem = np.array(self.last_memory[layer])

            #gradients for layer and iteration t
            weight_gradients = np.matmul(partial_derivitive_memory.T,mem)
            bias_gradients = np.sum(partial_derivitive_memory,axis=0)

            #update gradient decayed sums (exponentail average) and bias correct
            corrected_gradient_sums = {}
            self.gradient_sums["w"+str(layer)] = (self.decay)*self.gradient_sums["w"+str(layer)] + (1-self.decay)*np.square(weight_gradients)
            corrected_gradient_sums["w"+str(layer)] = self.gradient_sums["w"+str(layer)] / (1-np.power(self.decay,self.current_iteration))

            self.gradient_sums["b"+str(layer)] = (self.decay)*self.gradient_sums["b"+str(layer)] + (1-self.decay)*np.square(bias_gradients)
            corrected_gradient_sums["b"+str(layer)] = self.gradient_sums["b"+str(layer)] / (1-np.power(self.decay,self.current_iteration))

            #update gradient decayed momentums/velocity (exponentail average) and bias correct
            corrected_velocities = {}
            self.velocity["w"+str(layer)] = self.momentum*self.velocity["w"+str(layer)] + (1-self.momentum)*weight_gradients
            corrected_velocities["w"+str(layer)] = self.velocity["w"+str(layer)] / (1-np.power(self.decay,self.current_iteration))

            self.velocity["b"+str(layer)] = self.momentum*self.velocity["b"+str(layer)] + (1-self.momentum)*bias_gradients
            corrected_velocities["b"+str(layer)] = self.velocity["b"+str(layer)] / (1-np.power(self.decay,self.current_iteration))
            
            #update learning rates
            adapted_learning_rate_weights = (self.learning_rate)/(np.sqrt(corrected_gradient_sums["w"+str(layer)])+self.smoothing_rate)
            adapted_learning_rate_biases  = (self.learning_rate)/(np.sqrt(corrected_gradient_sums["b"+str(layer)])+self.smoothing_rate)

            #corrections to network values
            weight_corrections = adapted_learning_rate_weights*corrected_velocities["w"+str(layer)]
            bias_corrections = adapted_learning_rate_biases*corrected_velocities["b"+str(layer)]

            partial_derivitive_memory = np.dot(partial_derivitive_memory, self.network_values["w"+str(layer)])
            
            self.network_values["b"+str(layer)] -= bias_corrections
            self.network_values["w"+str(layer)] -= weight_corrections            
        return loss
        
    ##training
    ##  
    def train(self, trainX, trainY, batch_size=1, epochs=1, learning_rate_decay=0.0, shuffle=False):
        self.epoch_losses = []
        self.current_iteration = 0
        self.current_epoch = 0
        initial_learning_rate = self.learning_rate
        
        #start_time = time.time()
        for n in range(epochs):
            data = self.generate_batches(trainX, trainY, batch_size)
            iteration_losses = []
            self.current_epoch += 1
            for x, y in data:
                self.current_iteration += 1
                
                #self.back_propergate(x,y)
                #loss = self.adagrad(x,y)
                #loss = self.SGD(x,y)
                #loss = self.adadelta(x,y)
                loss = self.ADAM(x,y)

                iteration_losses.append(loss)
                
            self.learning_rate = self.learning_rate * (1/(1+learning_rate_decay*self.current_epoch))
            if shuffle==True:
                trainX, trainY = self.shuffle(trainX,trainY)

            print("Epoch: "+str(n+1)+"/"+str(epochs) + " complete")
            self.epoch_losses.append(iteration_losses)


            
        self.learning_rate = initial_learning_rate
        return np.array(self.epoch_losses)

    def test(self, testX, testY):
        correct = 0
        testxincorrect = []
        testyincorrect = []
        pred = []
        data = self.generate_batches(testX, testY, 1)
        
        for x, y in data:
            output = self.forward_propagate(x)
            if np.argmax(output) == np.argmax(y):
                correct += 1
            else:
                testxincorrect.append(x.reshape(28,28))
                testyincorrect.append(np.argmax(y))
                pred.append(np.argmax(output))
                
        return (correct, testxincorrect, testyincorrect, pred)
            
    @staticmethod
    def shuffle(trainX, trainY): #make take in *args
        combined = list(zip(trainX, trainY))
        random.shuffle(combined)
        uncombined = list(zip(*combined))
        return np.array(uncombined[0]), np.array(uncombined[1])
        
        
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
        network_values_str = {}
        for layer in self.network_values:
            network_values_str[str(layer)] = self.network_values[layer].copy().tolist()

        copy = [layer.copy() for layer in self.internal_architecture]
        for layer in copy:
            if "activation_class" in layer:
                layer.pop("activation_class")

        
        data = {"internal_architecture":copy, "learning_rate":self.learning_rate, "network_values":network_values_str}
        datadump = json.dumps(data)
        with open(network_name+".json", 'w+') as f:
            json.dump(data, f)
        print("Network exported")
        return datadump
    
    def import_network(self,json_network):
        pass

    @staticmethod
    def import_network(filename=None,json=None):
        import json
        if filename != None:
            with open(filename, 'r') as f:
                data = json.load(f)
        internal_architecture = data["internal_architecture"]
        learning_rate = data["learning_rate"]
        network = Network(internal_architecture)

        network_values = data["network_values"]
        for layer in network_values:
            network_values[str(layer)] = np.array(network_values[str(layer)])
        
        network.network_values = network_values
        print("Network imported")
        return network
    


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

def print_confusion_matrix(cls_true, cls_pred):
    assert len(cls_true) == len(cls_pred)
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.show()

def plot_losses(losses, average_line=True):
    figure, axis = plt.subplots(2, 2)

    x = range(len(losses.flatten()))
    y = losses.flatten()

    y1 = np.sum(losses, axis=1)/losses.shape[1]
    x1 = np.array(range(len(y1)))*losses.shape[1]+losses.shape[1]
    


    
    axis[0, 0].semilogy(x, y)
    axis[0, 0].semilogy(x1, y1)
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    axis[0, 0].semilogy(x, p(x))
    
    axis[0, 0].set_xlabel('iteration')
    axis[0, 0].set_ylabel('loss')
    axis[0, 0].set_title("All Losses vs Iterations (semilogy)")

    axis[0, 1].plot(x, y)
    axis[0, 1].plot(x1, y1)
    axis[0, 1].plot(x, p(x))
    axis[0, 1].set_xlabel('iteration')
    axis[0, 1].set_ylabel('loss')
    axis[0, 1].set_title("All Losses vs Iterations")
    
    axis[1, 1].semilogy(x, y)
    axis[1, 1].set_xlabel('iteration')
    axis[1, 1].set_ylabel('loss')
    axis[1, 1].set_title("Losses vs Iterations")
    
    axis[1, 0].semilogy(x1/losses.shape[1], y1)
    axis[1, 0].set_xlabel('iteration')
    axis[1, 0].set_ylabel('loss')
    axis[1, 0].set_title("Losses vs epochs")


    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.5, 
                    hspace=0.5)
    plt.show()







if __name__ == "__main__":

    internal_architecture = [
        {"inputNodes": 1},
        {"hiddenNodes": 2, "activation": 'sigmoid'},
        {"hiddenNodes": 2, "activation": 'sigmoid'},
        {"outputLayers": 1, "activation": 'leaky_RELU'}
    ]
    network = Network(internal_architecture)
    inputs = [[0],[0]]
    expected = [[1],[0]]

    print(sum(network.loss(network.forward_propagate(inputs), expected)))
    print(network.loss(network.forward_propagate(inputs), expected))
    print(network.network_values)
    losses = network.train(inputs,expected,1,5)
    losses.size
    print(network.network_values)
    print(sum(network.loss(network.forward_propagate(inputs), expected)))
    print(network.forward_propagate([1]))
    #network.export_network()
    

     












#####Notes and findings:
    #when the batch size without change of learning rate is increased the amount of change for a single item tended to over correcct, i.e large predictions of 4s and 8s. 
