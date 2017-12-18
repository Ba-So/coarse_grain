
import data_classes as dc

def test_Dyadic_mult():
    Test    = []

    data    = [2,3]
    A   = dc.Dyadic(data)
    B   = dc.Dyadic(data)
    A.multiply(B)
    out = 13
    Test.append(A.data == out)

    data    = [2,3]
    A   = dc.Dyadic(data)
    B   = 2
    A.multiply(B)
    out = [4,6]
    Test.append(A.data == out)

    data    = [[2,3],[2,3]]
    A   = dc.Dyadic(data)
    B   = dc.Dyadic(data)
    A.multiply(B)
    out = 4 + 9 + 2*6
    Test.append(A.data == out)

    data    = [[2,3],[2,3]]
    A   = dc.Dyadic(data)
    B   = 2
    A.multiply(B)
    out = [[4,6],[4,6]] 
    Test.append(A.data == out)
    
    if all( i == True for i in Test):
        return True
    else:
        return False

def test_Grid_avg():
    Test    =[]
    data    = [
                [[1,1],[1,1]],
                [[1,1],[1,1]]
                ] 
    A       = dc.Grid([2,2,2], data)
    A.avg(1)
    Test.append(A.data == data)
    data    = [
                [[1,2],[0,1]],
                [[1,3],[-1,1]]
                ] 

    A       = dc.Grid([2,2,2], data)
    A.avg(1)
    out    = [
                [[1,1],[1,1]],
                [[1,1],[1,1]]
                ] 
    Test.append(A.data == data)

    if all( i == True for i in Test):
        return True
    else:
        return False

if __name__== '__main__':
    print test_Dyadic_mult()
    print test_Grid_avg()
