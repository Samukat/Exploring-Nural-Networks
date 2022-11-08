from tkinter import *
import numpy as np

import matplotlib.pyplot as plt
from Network import Network, plot_images, print_confusion_matrix, plot_losses


class Drawer():
    size = 28
    pixel_size = 20
    
    def __init__(self, root):
        self.size = Drawer.size
        self.pixel_size = Drawer.pixel_size
        
        self.pixarray = np.zeros((28,28), dtype=int)
        self.network = Network.import_network("network_value_saves/smpled.json") ##importing a trained network
        self.preds = StringVar() ##label text for the predictions 
        self.preds.set("Network Predictions: \n\n\n ")
        
        self.root = root
        self.root.title = "Draw"
        self.root.geometry(f"{self.size*self.pixel_size+150}x{self.size*self.pixel_size}")
        self.root.configure(background="white")
        self.root.resizable(0,0)
        
        self.canvas = Canvas(self.root, bg='white', bd=5, height=self.size*self.pixel_size,width=self.size*self.pixel_size)
        self.canvas.place(x=150,y=0)
        self.setup()
        

    def setup(self):
        label = Label(self.root, textvariable=self.preds,justify="left",bg="white")
        label.place(relx = 0.0,
                 rely = 0.0,
                 anchor ='nw')


        
        self.canvas.bind("<B1-Motion>",self.mb1Motion)
        self.canvas.bind("<B3-Motion>",self.mb3Motion)

        self.canvas.bind("<Button-1>",self.mb1Motion)
        self.canvas.bind("<Button-3>",self.mb3Motion)

        self.canvas.bind("<Button-2>",self.clear)

    def mb1Motion(self,event):
        x1, y1 = (event.x),(event.y)
        x,y = x1//self.pixel_size, y1//self.pixel_size
        if (x>self.size-1 or x <0) or (y>self.size-1 or y<0):
            return

        self.pixarray[x,y] = 1
        self.canvas.create_rectangle(
            x*self.pixel_size,y*self.pixel_size,(x+1)*self.pixel_size,(y+1)*self.pixel_size,
            fill="black",
            outline=""
            )
        self.test()

    def mb3Motion(self,event):
        x1, y1 = (event.x),(event.y)
        x,y = x1//self.pixel_size, y1//self.pixel_size
        if (x>self.size-1 or x <0) or (y>self.size-1 or y<0):
            return
        
        self.pixarray[x,y] = 0
        self.canvas.create_rectangle(x*self.pixel_size,y*self.pixel_size,(x+1)*self.pixel_size,(y+1)*self.pixel_size, fill="white", outline="")

        self.test()
        
    def clear(self, event):
        self.pixarray = np.zeros((28,28), dtype=int)
        self.canvas.create_rectangle(0,0,self.size*self.pixel_size,self.size*self.pixel_size, fill="white", outline="")

        self.test()
             
    def test(self):
        flat_x = self.pixarray.flatten()
        output = self.network.forward_propagate(flat_x)
        text = "".join((f"{i}:  "+'{:f}'.format(b*100) + '\n') for i, b in enumerate(output[0]))
        text += "\n\nMax at:\n" + str(np.argmax(output[0]))


        self.preds.set("Network Predictions:\n\n\n"+text)



def start():
    root = Tk()
    d = Drawer(root)
    root.mainloop()
    
if __name__ == "__main__":
    start()
