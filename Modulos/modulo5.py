# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 12:25:15 2025

@author: Usuario
"""
import numpy as np
from modulo3 import norma

def columnas (A):
    res = []
    for i in range(0,A.shape[1]):
        columna = []
        for j in range (0,A.shape[0]):
            columna.append(A[j][i])
        res.append(columna)
    return res

def norma2(a):
    norma(a,2)

def multiplicacionMatricial (A,B):
    # Si A es un vector va a fallar .shape de numpy, por lo que lo convierto a matriz de 1 fila
    if len(A.shape) == 1:
        A = A.reshape(1, -1)
    # Lo mismo con B pero este solo puede ser un vector columna por lo que lo convierto a matriz de 1 columna
    if len(B.shape) == 1:
       B = B.reshape(-1, 1)
    if (A.shape[1] != B.shape[0]):
        return "No se puede calcular, dimensión incompatible"
    res=np.zeros((A.shape[0],B.shape[1]))
    #itero en las filas de A
    for l in range(0,A.shape[0]):
        #itero en las columnas de B para una misma fila de A
        for i in range(0,B.shape[1]):
            valorli = 0
            #calculo el valor de la posicion (l,i) multiplicando fila por columna
            for j in range(0,B.shape[0]):
                valorli += A[l,j]*B[j,i]
            res[l,i] = valorli
    #si el resultado no es una matriz sino un vector o valor devuelvo solo esa "fila" (primer elemento de la lista)
    if res.shape[0] == 1:
        return res[0]
    else:
        return res

def QRconGS (A,tol=1e-12, retornanops=False):
    #si no es cuadrada devuelve None
    if len(A.shape) == 1:
        if A.shape[0] != 1:
            return None 
    if A.shape[0] != A.shape[1]:
        return None
    ColumnasA = columnas(A)
    #Hago el algoritmo QR
    N_ops = 0
    Q = np.zeros((len(ColumnasA),len(ColumnasA)))
    R = np.zeros((len(ColumnasA),len(ColumnasA)))
    for j in range(0,len(ColumnasA)):
        a=ColumnasA[j]
        rjj = norma2(a)
        #paso a a vector
        a = np.array(a)
        #si rjj es practicamente 0 (menor a tol) le asigno 0
        if rjj < tol:
            R[j,j] = 0.0
            Q[0:len(ColumnasA), j] = 0.0
        else:
            R[j, j] = rjj
            qj = a / rjj
            Q[0:len(ColumnasA), j] = qj
        for i in range(j+1,len(ColumnasA)):
            rji = multiplicacionMatricial(qj,np.array(ColumnasA[i]))
            R[j,i] = rji
            ColumnasA[i] = ColumnasA[i] - rji*qj
            N_ops += 1
    print("Matriz Q:")
    for fila in Q:
        print(fila)
    
    print("\nMatriz R:")
    for fila in R:
        print(fila)
    if retornanops:
        print(N_ops)
#Ejemplo (da bien)
#A= np.array([[2,3],[0,4]])
#print (QRconGS(A, retornanops=True))

def QRconHH (A,tol=1e-12,retornanops = False):
    if len(A.shape) == 1:
        if A.shape[0] != 1:
            return None 
    if A.shape[0] < A.shape[1]:
        return None

metodos = ["RH","GS"]
def calculaQR (A, metodo="RH", tol=1e-12,retornanops = False):
    if metodo not in metodos:
        return None
    if metodo == "GS":
        return QRconGS(A,tol,retornanops)
    elif metodo == "RH":
        return QRconHH(A,tol,retornanops)