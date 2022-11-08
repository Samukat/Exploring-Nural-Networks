from tkinter import *
import numpy as np

class Drawer():
    size = 28
    pixel_size = 30
    
    def __init__(self, root):
        self.size = Drawer.size
        self.pixel_size = Drawer.pixel_size
        
        self.pixarray = np.zeros((28,28), dtype=int)
        
        self.root = root
        self.root.title = "Draw"
        self.root.geometry(f"{self.size*self.pixel_size}x{self.size*self.pixel_size}")
        self.root.configure(background="white")
        self.root.resizable(0,0)
        
        self.canvas = Canvas(self.root, bg='white', bd=5, height=self.size*self.pixel_size,width=self.size*self.pixel_size)
        self.canvas.place(x=0,y=0)
        self.setup()
        
        

    def setup(self):
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
        
        self.canvas.create_rectangle(x*self.pixel_size,y*self.pixel_size,(x+1)*self.pixel_size,(y+1)*self.pixel_size, fill="black", outline="")
        

    def mb3Motion(self,event):
        x1, y1 = (event.x),(event.y)
        x1, y1 = (event.x),(event.y)
        x , y = x1//self.pixel_size, y1//self.pixel_size
        
        if (x>self.size-1 or x <0) or (y>self.size-1 or y<0):
            return
        
        
        self.pixarray[x,y] = 0
        self.canvas.create_rectangle(x*self.pixel_size,y*self.pixel_size,(x+1)*self.pixel_size,(y+1)*self.pixel_size, fill="white", outline="")

    def clear(self, event):
        self.pixarray = np.zeros((28,28), dtype=int)
        self.canvas.create_rectangle(0,0,self.size*self.pixel_size,self.size*self.pixel_size, fill="white", outline="")

if __name__ == "__main__":
    root = Tk()
    d = Drawer(root)
    root.mainloop()
