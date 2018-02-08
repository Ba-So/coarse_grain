
import math_op as mo 
import numpy as np

class Grid(object):

    def __init__(self, dim, data, weights=1, vec=False):
        self.dim        = dim
        self.points(data, vec)
        # implement a rho array seperately
        # think about implementing Lat/lon grid seperately as well
        # can i guarantee that orders are not jumbled up?
        # thus the averaging is done independently from GRIDs 
        # done instead as function on grids. 
        
 
        
        # automatic weighting
        self.multiply(weights)

    
    def multiply(self, B):
        #multiplying a grid with another grid
        if type(B) is Grid:
            do.grid_mult(self, B)

    def points(self, data, vec):
        dim    = self.dim
        if vec:
            self.data = [[[Dyadic(data[i][j][k])
                            for k in range(dim[2])]
                                for j in range(dim[1])]
                                    for i in range(dim[0])]
          
        else:
            self.data = [[[data[i][j][k]
                            for k in range(dim[2])]
                                for j in range(dim[1])]
                                    for i in range(dim[0])]



class Dyadic(object):

    def __init__(self, data):
        self.dim    = len(data)
        self.data   = data 
        self.center = [lat,lon]


    def multiply(self, B):
        #multiplying this with another class Dyadic
        if type(B) is Dyadic:
            if type(B.data[0]) is list:
                self.data =  mo.tensor_mult(self, B)
            else:
                self.data =  mo.vector_mult(self, B)
        else:
            #assume a scalar
            if type(self.data[0]) is list:
                self.data   = [[self.data[k][j] * B
                                for j in range(len(self.data[0]))]
                                for k in range(len(self.data))]
            else:
                self.data   = [i*B for i in self.data]

    def add(self, B):
        if type(B) is Dyadic:
            return self.data + B.data

