import numpy as np


class Cell:
    def __init__(self,terrain=-1):
        """
        A basic cell of the board
        connectivity:                       

       / \     / \
     / 6 1 \ /     \
    |       |       |
    | 5   2 |       |
   / \ 4 3 / \     / \
 /     \ /     \ /     \
|       |       |       |
|       |       |       |
 \     / \     / \     /
   \ /     \ /     \ /
    |       |       |
    |       |       |
     \     / \     /
       \ /     \ /
                

colors:
0: river
1: red
2: yellow
3: blue
4: green
5: black
6: grey
7: brown

        """
        self._terrainType  = terrain
        self._connectSame  = 0
        self._connectivity = [-1] * 6
        
        return
    
    def __repr__(self):
        return repr(self._terrainType)
    
    def __getitem__(self): 
        return self._terrainType
        
    def connect(self,iface,terrain):
        self._connectivity[iface] = terrain
        self._connectSame = self._connectivity.count(self._terrainType)
        return
        
    def connectivity(self):
        return self._connectivity, self._connectSame
    
    def terrain(self):
        return self._terrainType

class Map:
    def __init__(self,nrows=9,maxrow=13,nwater=0,nterrains=7,maxodd=True):        
        self._nrows      = nrows
        self._maxrow    = maxrow
        self._size      = [nrows,maxrow]
        self._nwater    = nwater
        self._nterrains = nterrains
        self._cells     = np.empty((nrows, maxrow),dtype=object)

        # Initialize all to -1
        self._cells[:,:] = Cell(-1)

        # Set prohibited cells to -2
        if (maxodd):
            self._cells[1::2,0]  = Cell(-2)
            #self._cells[1::2,-1] = Cell(-2)
        else:
            self._cells[0::2,0]  = Cell(-2)
            #self._cells[0::2,-1] = Cell(-2)
        
        return

    def _cell(self,irow,icols):
        return self._cells[irow,icol]

    def _connectivity(self,icol,irow):
        
        if (0<=irow>=self._nrows):
            return [], -1
            
        return 0
    
    def size(self):
        return self._size
    
    def insert(self,irow,icol,terrain):
    
        if (self._cells[irow,icol].terrain() != -2.0):
            #self._cells[irow,icol] = terrain
            self._cells[irow,icol] = Cell(terrain)
            return 0
        else:
            return -1
        
    def cells(self):
        return self._cells[:,:]
        
    def cell(self,irow,icol):
        return self._cells[irow,icol].terrain()
    
    def generate(self):
    
        # 1. chose
    
        return


if __name__ =='__main__':
        
    from map_plot import *
    #from tkinter import *
        
    celula = Cell()
        
    celula.connect(1,1)
        
    mapa = Map(maxodd=False)
   
    mapa.insert(1,1,1)
    mapa.insert(1,4,1)
    mapa.insert(1,2,1)
    mapa.insert(0,1,1)
    
    print(mapa.cells())
              
    tk = Tk()

    grid = HexagonalGrid(tk, scale = 10, grid_width=mapa.size()[1], grid_height=mapa.size()[0])
    grid.grid(row=0, column=0, padx=2, pady=2)

    def correct_quit(tk):
        tk.destroy()
        tk.quit()

    quit = Button(tk, text = "Quit", command = lambda :correct_quit(tk))
    quit.grid(row=1, column=0)
    
    for i in range(mapa.size()[0]):
        for j in range(mapa.size()[1]):
            if(mapa.cell(i,j)>=0):
                grid.setCell(j,i, fill='blue')
            elif(mapa.cell(i,j)==-1):
                grid.setCell(j,i, fill='grey')
            elif(mapa.cell(i,j)==-2):
                grid.setCell(j,i, fill='white')
 
        
    tk.mainloop()
    
