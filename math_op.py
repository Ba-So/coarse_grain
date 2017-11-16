# all math Operations neccessary for computations
# I want to represent each of these as matrices. 
# what is more effective?
# memory wise: 
#           computing smaller matrices on the fly
#           I'd need gridsize times 3 matrices,
#           1 - rhov''v''
#           2 - delv
#           3 - v (to compute del v) so this is a throw away matrix
from itertools import cycle
import data_classes as dc

def bar_avg(area, position, data):
    """computes the other average"""
    avg_val = 0
    count   = 0 

    ydim    = len(data[0])
    xdim    = len(data[0][0])
    Dyadic  = isinstance(data[0][0][0], dc.Dyadic) 

    if Dyadic:
        avg_val = dc.Dyadic(data[0][0][0].dim)
        
    for j in range(0, area):

        for i in range(0, area):

            ypos    = position[1]-j
            xpos    = position[2]-i

            # for circularity:
            if ypos < 0:
                ypos    = ydim + ypos
            if xpos < 0:
                xpos    = xdim + xpos
            
            if Dyadic:
                avg_val.add(data[position[0]][ypos][xpos])
            else:
                avg_val += data[position[0]][ypos][xpos] 
            count   += 1

    # Norm
    if Dyadic:
        avg_val.multiply(1/count)
    else:
        avg_val /=  count 
        

    return avg_val 

    pass

def dyadic_product(A, B):
    """takes vectors"""
    pass
    # 

def tensor_mult(A, B):
    """computes the multiplication of two Tensors"""
    #maybe there are predefined functions somewhere
    C   = 0
    for i in range(A.dim):

        for j in range(A.dim):
            
            C   += A.data[i][j]*B.data[j][i]
            
    return C

def vector_mult(A,B):
    """computes the product of two vectors"""
    C   = 0

    for i in range(A.dim):

        C   += A.data[i]*B.data[i]

    return C

def grid_mult(A, B):
    '''multiplies a Dyadic (A) with a Dyadic or Skalar (B)'''
    if A.dim == B.dim:
        for i in range(A.dim[0]):
            for j in range(A.dim[1]):
                for k in range(A.dim[2]):
                    A.data[i][j][k].multiply(B.data[i][j][k])
                    







def partial_derivative(direc, quantity):
    """computes the partial derivative using differences"""
    # Carefull this has to take values from one Gridwindow next to it. 
    pass

def aerial_weighting():
    #computes areas for weighting.:H
    pass

