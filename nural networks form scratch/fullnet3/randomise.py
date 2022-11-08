from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import imutils


def function_timer(orgfunc):
    def wrapper_timer(*args, **kwargs):
        print("Before function {} was run".format(orgfunc.__name__))
        t = time.time()
        result = orgfunc(*args, **kwargs)
        print("{} took {:f}s to run".format(orgfunc.__name__, time.time()-t))
        return result
    return wrapper_timer

def scale(img, zoomfactor):
    h,w = img.shape
    M = cv2.getRotationMatrix2D( (w/2,h/2), 0, zoomfactor) 
    return cv2.warpAffine(img, M, img.shape[::-1])

#@function_timer
def randomise(img):
    #deviations 
    s = 0.2
    r = 20
    t = 3

    flat = False
    if img.shape[0] == 784:
        img = np.reshape(img,(28,28))
        flat = True
    

    
    randoms = np.random.rand(4)-0.5
    img = scale(img,1-randoms[3]*s*2)
    img = imutils.rotate(img, angle=randoms[0]*r*2)
    img = imutils.translate(img,randoms[1]*10,randoms[2]*t*2)

    if flat:
        return img.flatten()
    return img

###SLOWWWW PLS FIX
#@function_timer
def randomise_batch(batch):
    for i,img in enumerate(batch):
        batch[i] = randomise(img)
    return batch
    
    
    



if __name__ == "__main__":
    from Network import plot_images
    import time
    
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))
    #plot_images(train_X[0:9], train_y[0:9])

    #rdf = np.vectorize(randomise)
    #rdf = np.apply_along_axis(randomise, 2, train_X[0:9])
    #test = rdf(train_X[0:9])
    a = train_X[0:9]
    f = a.flatten().reshape(9,784)


    
    rdf = randomise_batch(f)
    rdf = np.reshape(rdf.flatten(),(9,28,28))

    plot_images(rdf, train_y[0:9])
    

    
    #for i in range(10):
     #   cv2.imshow("Translated", randomise(train_X[5]))
      #  cv2.waitKey(0)

