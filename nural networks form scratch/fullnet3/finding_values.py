from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from Network import Network, plot_images, print_confusion_matrix, plot_losses
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    internal_architecture = [
        {"inputNodes": 784},
        {"hiddenNodes": 128, "activation": "sigmoid"},
        {"hiddenNodes": 64, "activation": "sigmoid"},
        {"outputLayers": 10, "activation": "sigmoid"}
    ]

    #prep data train
    amount = 60000
    train_X = train_X[0:amount]
    train_y = train_y[0:amount]
    train_flat_x = train_X.flatten().reshape(amount,784)/255
    train_flat_y = np.zeros((amount,10), dtype=int)
    for i, value in enumerate(train_y):
            train_flat_y[i][value] = 1

    #prep data test
    amount = 10000
    test_X = test_X[0:amount]
    test_y = test_y[0:amount]
    test_flat_x = test_X.flatten().reshape(amount,784)/255
    test_flat_y = np.zeros((amount,10), dtype=int)
    for i, value in enumerate(test_y):
        test_flat_y[i][value] = 1

    
    values = 100
    test_value_a = np.random.uniform(low=0.0001, high=0.002, size=(values)) #lr
    test_value_b = np.random.uniform(low=0.8, high=0.99, size=(values)) #momentum
    test_value_c = np.random.uniform(low=0.9, high=0.999, size=(values)) #momentum
    results = np.zeros((values))

    plt.scatter(test_value_a,test_value_b)
    plt.show()
    
    for test in range(values):
        print("\nTest: "+str(test)+". Values {}, {}".format(test_value_a[test],test_value_b[test]))
        network = Network(internal_architecture, learning_rate=test_value_a[test], momentum=test_value_b[test],decay=0.99)
        network.train(train_flat_x,train_flat_y,32,1, shuffle=False)
        results[test] = network.test(test_flat_x,test_flat_y)[0]/10000
        
    print(results)

    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(test_value_a,test_value_b, c=results, cmap=cm)
    plt.colorbar(sc)
    plt.show()

    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.tricontourf(test_value_a, test_value_b, results, 30)
    plt.colorbar(sc)
    plt.show()
        


        
    


