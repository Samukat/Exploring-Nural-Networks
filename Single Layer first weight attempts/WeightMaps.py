import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.keras.datasets import mnist




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
    
def evaluate_image(image, weights):
    maxcat = None
    maxval = 0
    for numcat in weights:
        value = numcat.weigh_image(image)
        if numcat == None:
            maxcat = numcat
            continue
        
        if value > maxval:
            maxcat = numcat
            maxval = value

    return (maxcat, maxval)

class NumCat():
    def __init__(self, num, weight=None):
        self.num = num
        self.editions_weight = 0
        self.bias = 0
        self.negative_weight_multiplier = 1.5
        if weight == None:
            self.weight_map = np.zeros((784,)) #np.zeros((784,), dtype=int)
        else:
            self.weight_map = weight

    def get_normalised_weight(self, editions_weight=None):
        #print(np.amax(self.weight_map / self.editions_weight))
        #return self.weight_map / self.editions_weight


        return self.weight_map / np.amax(self.weight_map)

    def add_weight(self, image):
        self.weight_map = np.add(self.weight_map, image.ravel())
        self.editions_weight += 1
        return self.weight_map

    def subtract_weight(self, image):
        self.weight_map = np.subtract(self.weight_map, image.ravel()*self.negative_weight_multiplier)
        
        #self.editions_weight += 1
        return self.weight_map

    def weigh_image(self, image):
        image = image.ravel()
        mult = np.multiply(self.get_normalised_weight(), image)
        out = mult + self.bias
        return np.sum(out)
        
    def __int__(self):
        return self.num
        




#train_y = to_categorical(train_y)
#ata = MNIST(data_dir="data/MNIST/")


(train_X, train_y), (test_X, test_y) = mnist.load_data()
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

#print(type(test_X))

plot_images(train_X[0:9],train_y[0:9])






learning_iterations = 30000
negative_weight_iterations = 60000
bias_iterations = 20000

numcats = {}
for i, training_image in enumerate(train_X[0:learning_iterations]):
    if train_y[i] not in numcats:
        numcats[train_y[i]] = NumCat(train_y[i])         #numcats[train_y[i]] = np.zeros((784,), dtype=int)
    numcats[train_y[i]].add_weight(training_image)
plot_images(list(a.get_normalised_weight().reshape(28,28) for a in numcats.values())[0:9], [0,1,2,3,4,5,6,7,8])
          
for i, training_image in enumerate(train_X[0:negative_weight_iterations]):
    eva = evaluate_image(train_X[i],numcats.values())
    if (eva[0].num != train_y[i]):
        numcats[eva[0].num].subtract_weight(train_X[i])
        numcats[train_y[i]].add_weight(train_X[i])
plot_images(list(a.get_normalised_weight().reshape(28,28) for a in numcats.values())[0:9], [0,1,2,3,4,5,6,7,8])

bais_weight = 0.001
for o in range(1):
    print(bais_weight/(o+1))
    for i, training_image in enumerate(train_X[0:bias_iterations]):
        eva = evaluate_image(train_X[i],numcats.values())
        if (eva[0].num != train_y[i]):
            numcats[train_y[i]].bias += bais_weight/(o+1)



for key, value in numcats.items():
    print(key, value.bias)
        





Output_test = np.array(1,)
tests= 10000
correct = 0
incorrect = 0


###Tests 
wrongimg = []
wrongpred = []
wrongnum = []


y_pred = []
for test in range(tests):
    eva = evaluate_image(test_X[test],numcats.values())
    #print(eva[0].num, test_y[test])
    y_pred.append(eva[0].num)
    if (eva[0].num == test_y[test]):
        correct += 1
    else:
        wrongimg.append(test_X[test])
        wrongnum.append(test_y[test])
        wrongpred.append(eva[0].num)
        
        

print("accuracy = {}".format(correct/tests))
print_confusion_matrix(test_y[0:tests], y_pred)

off = 10
plot_images(wrongimg[0+off:9+off],wrongnum[0+off:9+off],wrongpred[0+off:9+off])
#for numcat in numcats.values():
    #print(numcat.weigh_image(test_X[0]), numcat.num)



#print(numcats[4].weight_map)




#plt.imshow(numcats[6].get_normalised_weight().reshape(28,28))
#plt.show()



