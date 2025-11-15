# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 18:08:26 2025

@author: Usuario
"""
import numpy as np
#funcion que devuelve el valor absoluto de un numero
def absoluto (x):
    if x < 0:
        return -x
    else:
        return x

#funcion que devuelve el error absoluto de un numero y su version en numero de maquina
def error(x,y):
    y= np.float64(y) #me aseguro de que y sea de tipo float64 (numero de maquina)
    return absoluto(x-y)

#funcion que devuelve el error relativo de un numero y su version en numero de maquina
def error_relativo (x,y):
    return (error(x,y)/absoluto(x))


#funcion que devuelve True si 2 matrices A y B son iguales (asumo que se puede usar shape)
def matricesIguales(A,B,tol=1e-8):
    if A.shape != B.shape:
        return False
    else:
        for i in range (A.shape[0]):
            for j in range (A.shape[1]):
                #agrego tolerancia ya que sino no pasa los tests porque me quedan valores con numero de maquina
                if error(A[i,j], B[i,j]) > tol:
                    return False
    return True


