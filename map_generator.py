# -*- coding: utf-8 -*-
"""
Created on Fri May 06 10:08:26 2016

@author: Ajoo
"""

import numpy as np
import random

from map_plot import plot_hex

from collections import deque
from itertools import combinations_with_replacement
import copy

#TODO implement a stochastic queue with a given pdf of popping according to position
class SearchAgent(object):
    def __init__(self, queue=deque):
        self.queue = queue()
        
    def search(self, node):
        while not node.is_goal:
            self.queue.extend(node.children)
            try:
                node = self.queue.pop()
            except IndexError:
                return None
        return node

class SearchNode(object):
    #TODO: make this an interface
    pass

class RiverSearchNode(SearchNode):
    '''
    Node in the search for river solutions. Stores an internal state for the 
    search
    '''
    def __init__(self, board, nriver, sample_directions=[-1,0,1], end_in_river=True):
        self.board = board
        self.nriver = nriver
        self.sample_directions = sample_directions
        self.end_in_river = end_in_river
    
    @staticmethod
    def start(board, nriver, nspawns=2, pad=2, sample_directions=[-1,0,1], end_in_river=True):
        start = RiverSearchNode(board, nriver, sample_directions)
        start.river = set(zip(*np.nonzero(board.tiles)))
        
        start.xlist = []

        for i in range(nspawns):
            done = False
            while not done:
                x, a = start.board.random_border_and_direction()
                if x not in start.river:
                    start.xlist.append((x,a))
                    start.river.add(x)
                    done = True
                
        return start
      
    @property
    def is_goal(self):
        return not self.xlist and len(self.river) == self.nriver
    
    @property
    def children(self):
        children = []
        
        if not self.xlist:
            return children
            
        for dalist in combinations_with_replacement(self.sample_directions, len(self.xlist)):
            child = self.copy()
            for xa, da in zip(self.xlist, dalist):
                x, a = xa
                x = self.board.neighbourhood(x)[a]
                a = np.mod(a + da, 6)

                if self.board.in_bounds(x, 1) and (x not in child.river or not self.end_in_river):
                    child.xlist.append((x, a))
                child.river.add(x)
            if len(child.river) < self.nriver or not child.xlist:
                children.append(child)
        
        random.shuffle(children)
        return children
        
    def copy(self):
        c = copy.copy(self)
        c.xlist = []
        c.river = c.river.copy()
        return c
        
class TerrainSearchNode:
    '''
    Node in the search for terrain solutions. Stores an internal state for the 
    search
    '''
    def __init__(self):
        pass
        
    @staticmethod
    def start(board, nterrains):
        start = TerrainSearchNode()
        start.board = board.copy()
        ntiles = np.sum(board.tiles == 0)
        if not ntiles % nterrains == 0:
            raise Exception("Number of assignable tiles not a multiple of the number of terrains")
        
        #maintain counts of number of terrain assignements so we don't have to consistently recount them
        start.counts = (ntiles//nterrains)*np.ones((nterrains,), dtype=np.int)
        start._cursor = np.argmax(board.tiles.ravel()==0)
        return start
        
    @property
    def is_goal(self):
        return np.all(self.counts == 0)
    
    @property
    def cursor(self):
        return self._cursor//self.board.width, self._cursor%self.board.width
    
    @property
    def next_cursor(self):
        if self._cursor == self.board.tiles.size-1:
            return None
            
        return self._cursor + np.argmax(self.board.tiles.ravel()[self._cursor+1:]==0)+1
            
    
    @property
    def children(self):
        #self.board.plot()
        if self._cursor is None:
            return []
            
        next_cursor = self.next_cursor
        children = [self.child(i, next_cursor) for i in range(self.nterrains)\
            if self.counts[i] > 0 and i+2 not in self.board.neighbour_tiles(self.cursor)]        
        
        random.shuffle(children)
        return children
        
    @property
    def nterrains(self):
        return len(self.counts)
        
    def child(self, i, next_cursor):
        c = copy.copy(self)
        c.counts = c.counts.copy()
        c.counts[i] -= 1
        c.board = c.board.copy()
        c.board[c.cursor] = i+2
        c._cursor = next_cursor
        return c


class Board(object):
    '''
    Hexagonal tile board. Tiles are stored in odd-row horizontal offset coordinates
    '''
    def __init__(self, height=9, width=13, nterrains=7, nriver=36, nspawns=(1,1)):
        #nterrains += 1
        #self.tiles = np.random.randint(0, nterrains, (height, width))
        #self.tiles = np.mod(np.arange(height*width, dtype=np.int), nterrains).reshape((height, width))
        self.tiles = np.zeros((height, width), dtype=np.int)
        
        while True:
            for i in range(nspawns[0]):
                self.generate_river()       #generate nspawn[0] unconstrained rivers
            if np.sum(self.tiles) < nriver - 3*nspawns[1]:  #reserve at least 3 tiles per constrained river
                break
        
        self.generate_rivers_constrained(nriver, nspawns[1])    #generate nspawn[1] constrained rivers
        
        self.set_outside_tiles()
        self.generate_terrain(nterrains)
        
    def random_border_and_direction(self, pad=2):
        '''
        Pick a random point in the border and a "prependicular" direction
        '''
        #This implementation makes it equaly likely to pick any of the 4 sides except corners
        r = np.random.rand()
        if r < .5:
            i = int(r < .25)*(self.height-1)
            j = np.random.randint(pad, self.width-pad-1)
            a = (1 if i == 0 else 4) + int(r < .125 or r >= .375)
            return (i,j), a
        else:
            i = np.random.randint(pad, self.height-pad-1)
            j = int(r < .75)*(self.width-1-(i&1))
            a = 0 if j == 0 else 3
            return (i,j), a

    def set_outside_tiles(self):
        self.tiles[1::2, self.width-1] = -1
            
    def set_river(self, river):
        self.tiles[:,:] = 0
        self.tiles[tuple(zip(*list(river)))] = 1
    
    def generate_rivers_constrained(self, nriver, nspawns=2, pad=2, sample_directions=[-1,0,1], end_in_water=True):
        '''
        Generates nspawns rivers taking into account that the total number of 
        river tiles has to be exactly nriver
        '''
        result = False
        while not result: #if search fails try from a new start node
            start = RiverSearchNode.start(self, nriver, nspawns, pad, sample_directions, end_in_water)
            result = SearchAgent().search(start)
        
        self.set_river(result.river)
    
    def generate_river(self, pad=2, sample_directions = lambda : np.random.randint(-1,2)):
        '''
        Generates a single river without caring how many river tiles it takes
        '''
        x, a = self.random_border_and_direction()
        self[x] = 1
        
        x = self.neighbourhood(x)[a]
        self[x] = 1
        
        while self.in_bounds(x, 1):
            #self[x] = 1
            a = np.mod(a + sample_directions(), 6)
            x = self.neighbourhood(x)[a]
            self[x] = 1
        
    def generate_terrain(self, nterrains):
        start = TerrainSearchNode.start(self, nterrains)
        result = SearchAgent().search(start)
        
        self.tiles = result.board.tiles

    def plot(self):
        plot_hex(np.maximum(self.tiles, 0))
#        plt.imshow(self.tiles, cmap=plt.cm.gray)

    @staticmethod
    def cube2oddr(xyz):
        return xyz[0], int(xyz[2] + (xyz[0] - (xyz[0]&1))/2)
        
    def __getitem__(self, index):
        if len(index) == 3: #cube coordinates
            index = Board.cube2oddr(index)    
        return self.tiles.__getitem__(index)
        
    def __setitem__(self, index, value):
        if len(index) == 3: #cube coordinates
            index = Board.cube2oddr(index)    
        return self.tiles.__setitem__(index, value)
        
    
    def in_bounds(self, index, offset=0):
        return index[0]>=offset and index[1]>=offset and \
            index[0]<self.height-offset and index[1]+(index[0]&1)<self.width-offset
            
    def in_border(self, index):
        return self.in_bounds(index, offset=1)

    neighbours = (((0,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0)),
                  ((0,1), (1,1), (1,0), (0,-1), (-1,0), (-1,1)))    
    def neighbourhood(self, index):
        if len(index) == 3:
            index = Board.cube2oddr(index)
        parity = index[0]&1
        index = [(index[0]+d[0], index[1]+d[1]) for d in Board.neighbours[parity]]
        return index
        
    def neighbour_tiles(self, index):
        index = [idx for idx in self.neighbourhood(index) if self.in_bounds(idx)]
        return self.tiles[tuple(map(list, zip(*index)))]
        
    def copy(self):
        c = copy.copy(self)
        c.tiles = c.tiles.copy()
        return c
                
    @property
    def height(self):
        return self.tiles.shape[0]
        
    @property
    def width(self):
        return self.tiles.shape[1]
        
if __name__ == "__main__":
    b = Board(nriver=36, nspawns=(3,1))
    print(np.sum(b.tiles==1))
    b.plot()
    