import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from tensorflow.keras.datasets import cifar10

##attempting to get filters for CNN working

(train_X, train_y), (test_X, test_y) = cifar10.load_data()
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))


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


filter_grid = np.array([[-1,-1,-1],
                        [-1,8,-1],
                        [-1,-1,-1]])
image = train_X[8]

f, axarr = plt.subplots(4,3)



#plain image
axarr[0,0].imshow(image)

#black/white
bw_image = np.sum(image,axis=2)/3
axarr[0,1].imshow(bw_image)

#take part
x,y = 5,26
part = image[y-1:y+2,x-1:x+2]
part2 = bw_image[y-1:y+2,x-1:x+2]
axarr[1,0].imshow(part)
axarr[1,1].imshow(part2)
axarr[1,2].imshow(filter_grid)

#apply filter
filtered = np.multiply(filter_grid,np.transpose(part, axes=[2,0,1]))
filtered = np.transpose(filtered, axes=[1,2,0])

filtered2 = np.multiply(filter_grid,part2)
axarr[2,0].imshow((filtered+255)//2)
axarr[2,1].imshow(filtered2)

#sum filtered
dot = np.ones((1,1,3), np.int32)
dot[0][0] = np.sum(filtered,axis=(0,1))
dot1 = np.ones((1,1), np.int32)
dot1[0][0] = np.sum(filtered2,axis=(0,1))

axarr[3,0].imshow(np.atleast_3d(dot+255)//18)
axarr[3,1].imshow(np.atleast_2d(dot1))
plt.show()





def convolute1(image,kernal): #kinda works
    new_shape = list(image.shape)
    new_shape[0] -= 2
    new_shape[1] -= 2
    new_image = np.zeros(new_shape)
    
    for x in range(1, image.shape[0]-2):
        for y in range(1, image.shape[1]-2):
            part = image[y-1:y+2,x-1:x+2]
            filtered = np.multiply(filter_grid,np.transpose(part, axes=[2,0,1]))
            filtered = np.transpose(filtered, axes=[1,2,0])
            new_image[y,x] = np.sum(filtered,axis=(0,1))
    return new_image


def convolute2(image,kernal): #seperating RGB beforehand
    new_shape = image.shape
    if image.shape == 3:
        RGB_image = np.transpose(image, axes=[2,0,1])
        new_image = np.zeros(new_shape)
    else:
        BW_image = np.array([image])
        new_image = np.array([np.zeros(new_shape)])


                                 
    
    image[0] -= 2
    image[1] -= 2
    
    
    for x in range(1, image.shape[0]-2):
        for y in range(1, image.shape[1]-2):
            part = image[y-1:y+2,x-1:x+2]
            filtered = np.multiply(filter_grid,np.transpose(part, axes=[2,0,1]))
            filtered = np.transpose(filtered, axes=[1,2,0])
            new_image[y,x] = np.sum(filtered,axis=(0,1))
    return new_image

print(convolute1(image,filter_grid))
f, axarr = plt.subplots(1,2)
axarr[0].imshow(image)
axarr[1].imshow((convolute1(image,filter_grid)+255*8)/(255*16))
#plt.imshow((convolute(image,filter_grid)+255)/(255*9*2))
plt.show()
















            
            
