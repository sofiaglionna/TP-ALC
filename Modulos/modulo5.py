# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 12:25:15 2025

@author: Usuario
"""
import numpy as np
from modulo3 import norma
from AUXILIARES import traspuestaConNumpy,multiplicacionMatricialConNumpy

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
            rji = multiplicacionMatricialConNumpy(qj,np.array(ColumnasA[i]))
            if rji.shape[0] == 1:
                rji = rji[0]
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
    
#No usamos el algoritmo de la guia, usamos uno mejor optimizado (no construye H moño, muy costoso en matrices A grandes)
def QR_con_HH (A,tol=1e-12):
    if len(A.shape) == 1:
        if A.shape[0] != 1:
            return None 
    if A.shape[0] < A.shape[1]:
        return None
    filas, columnas = A.shape
    
    R = A.copy()
    #paso a float para evitar errores
    R= R.astype(float)
    Q = np.identity(filas)
    #No calculo todo Q y R sino sus versiones reducidas para optimizar
    for k in range(0, min(filas, columnas)):
        X = R[k:filas,k].copy()
        a= (signo(X[0]))*(norma2(X))
        u = X - (a*canonico(0,filas-k))
        if norma2(u) > tol:
            u = u/norma2(u)
            R_sub = R[k:filas, 0:columnas]
            UporR = multiplicacionMatricialConNumpy(u, R_sub)
            if UporR.shape[0] == 1:
                UporR = UporR[0]
            #UporR siempre es un vector ya que u lo es y es un producto matricial
            R[k:filas, 0:columnas] = R_sub - 2*productoExterior(u,UporR)
            Q_sub = Q[0:filas, k:filas]
            Qu = multiplicacionMatricialConNumpy(Q_sub, u)
            if Qu.shape[0] == 1:
                Qu = Qu[0]
            #Qu siempre es un vector columna, lo paso a vector fila para producto exterior. (tomo la posicion 0 ya que traspuesta devuelve una matriz (1xlen(Qu))y yo quiero un vector)
            Q[0:filas, k:filas] = Q_sub - 2*productoExterior(traspuestaConNumpy(Qu)[0], u)
    return Q,R

metodos = ["RH","GS"]
def calculaQR (A, metodo="RH", tol=1e-12,retornanops = False):
    if metodo not in metodos:
        return None
    if metodo == "GS":
        return QR_con_GS(A,tol,retornanops)
    elif metodo == "RH":
        return QR_con_HH(A,tol)
