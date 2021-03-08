from debugdecorators import TimeThisNew,PrintArgs,PrintReturn

@TimeThisNew
@PrintReturn
@PrintArgs
def test_func(a,b):
    c = 0
    for i in range(a):
        for j in range(b*i):
            c = c + i*j
    return c

if __name__ == '__main__':
    a = test_func(100,10000)
    print('result is {}'.format(a))

