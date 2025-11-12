# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 12:25:15 2025

@author: Usuario
"""
import numpy as np
from modulo3 import norma

#toma matriz con numpy y devuelve matriz con numpy
def traspuestaConNumpy (A):   
    res = []
    #si es un vector
    if len(A.shape) == 1:
        for i in range(0,A.shape[0]):
            res.append([A[i]])
        return np.array(res)
    for i in range(0,A.shape[1]):
        columna = []
        for j in range(0,A.shape[0]):
            columna.append(A[j][i])
        res.append(columna)
    #si res tiene solo 1 fila devuelvo esa sola sin forma de matriz (esto pasa cuando A es un vector columna)
    if len(res) ==1:
        return np.array(res[0])
    return np.array(res)

def columnas (A):
    res = []
    for i in range(0,A.shape[1]):
        columna = []
        for j in range (0,A.shape[0]):
            columna.append(A[j][i])
        res.append(columna)
    return res

def norma2(a):
    return norma(a,2)

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

def QR_con_GS (A,tol=1e-12, retornanops=False):
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
        #len(a) multiplicaciones, len(a)-1 sumas y una raiz cuadrada
        N_ops += 2*len(a)
        rjj = norma2(a)
        #paso a a vector
        a = np.array(a)
        #si rjj es practicamente 0 (menor a tol) le asigno 0
        if rjj < tol:
            R[j,j] = 0.0
            Q[0:len(ColumnasA), j] = 0.0
        else:
            R[j, j] = rjj
            #len(a) divisiones 
            qj = a / rjj
            N_ops += len(a)
            Q[0:len(ColumnasA), j] = qj
        for i in range(j+1,len(ColumnasA)):
            # len(qj) multiplicaciones y sumas -> 2*len(qj) operaciones
            rji = multiplicacionMatricial(qj,np.array(ColumnasA[i]))
            R[j,i] = rji
            # len(qj) multiplicaciones y restas -> 2*len(qj) operaciones
            ColumnasA[i] = ColumnasA[i] - rji*qj
            #cuento operaciones
            N_ops += 2*len(qj)
            N_ops += 2*len(qj)
    #print para visualizar
    #print("Matriz Q:")
    #for fila in Q:
    #    print(fila)
    #
    #print("\nMatriz R:")
    #for fila in R:
    #    print(fila)
    if retornanops:
    #    print(N_ops)
        return Q,R,N_ops
    else:
        return Q,R
#Ejemplo (da bien)
#A= np.array([[2,3],[0,4]])
#print (QRconGS(A, retornanops=True))

#devuelve 1 si k es positivo, -1 si es negativo
def signo (k):
    if k<0:
        return (-1)
    else: 
        return 1
#devuelve el vector canonico con 1 en i y dimension dim
def canonico (i,dim):
    res = np.zeros(dim)
    res[i] = 1
    return res

#funcion que calcula el productoExterior entre 2 vectores fila (toma como si el primero (A) fuera columna)
def productoExterior (A,B):
    res = np.zeros((len(A),len(B)))
    for i in range (0,len(A)):
        for j in range (0,len(B)):
            res[i][j] = A[i]*B[j]
    return res
    
def QR_con_HH (A,tol=1e-12):
    if len(A.shape) == 1:
        if A.shape[0] != 1:
            return None 
    if A.shape[0] < A.shape[1]:
        return None
    R = A.copy()
    Q = np.identity(A.shape[0])
    for k in range(0,A.shape[1]):
        X = R[k:A.shape[0],k].copy()
        a= (signo(X[0]))*(norma2(X))
        u = X - (a*canonico(0,A.shape[0]-k))
        if norma2(u) > tol:
            u = u/norma2(u)
            H = np.identity(A.shape[0]-k) - 2*productoExterior(u,u)
            H_moño = np.identity((A.shape[0]))
            H_moño[k:A.shape[0],k:A.shape[0]] = H
            R = multiplicacionMatricial(H_moño, R)
            Q = multiplicacionMatricial(Q,traspuestaConNumpy(H_moño))
    return Q,R

#Ejemplo (da bien)
#A= np.array([[2,3],[0,4]])
#print (QRconHH(A))

metodos = ["RH","GS"]
def calculaQR (A, metodo="RH", tol=1e-12,retornanops = False):
    if metodo not in metodos:
        return None
    if metodo == "GS":
        return QR_con_GS(A,tol,retornanops)
    elif metodo == "RH":
        return QR_con_HH(A,tol)
