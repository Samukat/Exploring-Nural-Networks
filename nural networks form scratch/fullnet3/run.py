from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from Network import Network, plot_images, print_confusion_matrix, plot_losses
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))
    #plot_images(train_X[0:9], train_y[0:9])

    ##Setup
    internal_architecture = [
        {"inputNodes": 784},
        {"hiddenNodes": 128, "activation": "sigmoid"},
        {"hiddenNodes": 64, "activation": "sigmoid"},
        {"outputLayers": 10, "activation": "sigmoid"}
    ]
    #network = Network(internal_architecture, learning_rate=0.03, momentum=0.55)
    #network = Network(internal_architecture, learning_rate=0.03, momentum=0.0,decay=0.6)
    network = Network(internal_architecture, learning_rate=0.001, momentum=0.9,decay=0.99)
    #network.export_network()
    #network = Network.import_network("9808.json")
    if True:
        #Train
        amount = 60000
        train_X = train_X[0:amount]
        train_y = train_y[0:amount]
        flat_x = train_X.flatten().reshape(amount,784)/255
        flat_y = np.zeros((amount,10), dtype=int)
        for i, value in enumerate(train_y):
            flat_y[i][value] = 1
        losses = network.train(flat_x,flat_y,32,6, shuffle=True) #https://arxiv.org/pdf/1711.00489.pdf
        plot_losses(losses)
        
    
        

        #network.export_network()

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
    print_confusion_matrix(test_results[2],test_results[3])
    
