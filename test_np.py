#!/usr/bin/env python
# coding=utf-8
import numpy as np


def test_npmult():
   A = np.arange(81).reshape([3,3,3,3])
   B = np.arange(27).reshape([3,3,3])

   C_loop = np.empty([3,3,3,3])
   C_np = np.empty([3,3,3,3])
   for i in range(3):
       for j in range(3):
           for k in range(3):
               for l in range(3):
                   C_loop[j, i,  k, l] = A[j, i,  k, l] * B[j, k, l]

   C_np = np.multiply(A, B)

   print np.all(C_np == C_loop)
   return None

def test_npmult_vec():
    A = np.arange(27).reshape([3,3,3])
    B = np.arange(3)
    c0 = np.empty([3, 3, 3])
    c1 = np.empty([3, 3, 3])
    c2 = np.empty([3, 3, 3])
    for i in range(3):
        c0[i, :, :] = A[i, :, :] * B[i]
        c1[:, i, :] = A[:, i, :] * B[i]
        c2[:, :, i] = A[:, :, i] * B[i]
    d0 = np.multiply(A, B[:, np.newaxis, np.newaxis])
    d1 = np.multiply(A, B[np.newaxis, :, np.newaxis])
    d2 = np.multiply(A, B[np.newaxis, np.newaxis, :])
    print np.all(c0 == d0)
    print np.all(c1 == d1)
    print np.all(c2 == d2)

    return None


def test_npeinsum_three():
   A = np.arange(81).reshape([3,3,3,3])
   B = np.arange(81).reshape([3,3,3,3])
   C = np.arange(27).reshape([3,3,3])

   C_loop = np.empty([3,3])
   C_np = np.empty([3,3,3,3])
   for i in range(3):
       for j in range(3):
           help = np.multiply(A[i, j, :, :], B[i, j, :, :])
           C_loop[:, :] += np.multiply(help, C[j, : , :])

   C_np = np.einsum('ijkl,ijkl,jkl->kl', A, B, C)
   print C_np
   print ('__')
   print C_loop

   print np.all(C_np == C_loop)
   return None

def test_npeinsum():
   A = np.arange(81).reshape([3,3,3,3])
   B = np.arange(81).reshape([3,3,3,3])

   C_loop = np.empty([3,3,3,3])
   C_np = np.empty([3,3,3,3])
   for i in range(3):
       for j in range(3):
           C_loop[:, :] += np.multiply(A[i, j,  :, :], B[i, j, :, :])

   C_np = np.einsum('ijkl,ijkl->kl', A, B)

   print np.all(C_np == C_loop)
   return None

def test_npsum():
   A = np.arange(81).reshape([3,3,3,3])
   B = np.arange(27).reshape([3,3,3])

   C_loop = np.empty([3,3,3,3])
   C_np = np.empty([3,3,3,3])
   for i in range(3):
       C_loop[:, i, :, :] = np.add(A[:, i, :, :], B[:, :, :])

   C_np = np.add(A, B[:, np.newaxis, :, :])

   print np.all(C_np == C_loop)
   return None

def test_npeinsum_list():
   A = np.arange(27).reshape([3,3,3])
   C = [A,A]
   C = np.array(C)

   C_loop = np.empty([2,2,3,3,3])
   C_loop_2 = np.empty([2,2,3,3,3])
   for i in range(2):
       for j in range(2):
           C_loop[i,j,:,:,:] = np.multiply(C[i,:,:,:], C[j,:,:,:])

   for i in range(2):
       for j in range(2):
           C_loop_2[i,j,:,:,:] = C[i,:,:,:] * C[j,:,:,:]

   C_np = np.einsum('jklm,iklm->ijklm', C, C)
   print C_np
   print C_loop

   print np.all(C_np == C_loop)
   print np.all(C_loop_2 == C_loop)

   return None

if __name__ == '__main__':
    test_npmult()
    test_npeinsum()
    test_npsum()
    test_npmult_vec()
    test_npeinsum_list()
    test_npeinsum_three()
