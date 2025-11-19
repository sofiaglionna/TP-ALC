# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 12:25:15 2025

@author: Usuario
"""
import numpy as np
from modulo3 import norma
from AUXILIARES import traspuestaConNumpy,multiplicacionMatricialConNumpy,traspuestaFilaACol,producto_interno

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

def QR_con_GS_Para_TP (A,tol=1e-12, retornanops=False):
    N_ops = 0
    filasA,columnasA = A.shape
    Q=np.zeros(A.shape)
    R=np.zeros((columnasA,columnasA))
    Atraspuesta = traspuestaConNumpy(A)
    N_ops += 2*filasA + 1  # costo de la norma (filasA multiplicaciones y sumas y una raiz cuadrada al final)
    norma2DeTransA = norma2(Atraspuesta[0])
    N_ops += filasA  #costo de divisiones
    Q[:,0]=Atraspuesta[0]/norma2DeTransA
    R[0,0] = norma2DeTransA
    for j in range (1,columnasA):
        Qj = Atraspuesta[j]
        for k in range (0,j):
            columnaQk = Q[:,k]
            N_ops += 2*filasA #costo de multiplicacion matricial entre 2 vectores
            #como ambos son vectores fila hago producto interno
            rkj = producto_interno(columnaQk,(Qj))
            R[k,j] = rkj
            N_ops += 2*filasA #numero filasA de multiplicaciones y restas
            Qj = Qj - rkj*columnaQk
        N_ops += 2*filasA + 1  # costo de la norma
        R[j,j] = norma2(Qj)
        #si rjj es practicamente 0 asigno 0 a la columna qj (a mayor rjj mas independiente es Qj, si es 0 es LD de los anteriores)
        if R[j,j] < tol:
            Q[:, j] = 0
            continue
        N_ops += filasA #filasA divisiones
        Q[:,j] = (Qj/R[j,j])
    
    if retornanops:
        return Q,R,N_ops
    else:
        return Q,R

#funciona para A con #filas>=#columnas
def QR_con_GS (A,tol=1e-12, retornanops=False):
    if A.shape[0] != A.shape[1]:
        return None
    N_ops = 0
    filasA,columnasA = A.shape
    Q=np.zeros(A.shape)
    R=np.zeros((columnasA,columnasA))
    Atraspuesta = traspuestaConNumpy(A)
    N_ops += 2*filasA + 1  # costo de la norma (filasA multiplicaciones y sumas y una raiz cuadrada al final)
    norma2DeTransA = norma2(Atraspuesta[0])
    N_ops += filasA  #costo de divisiones
    Q[:,0]=Atraspuesta[0]/norma2DeTransA
    R[0,0] = norma2DeTransA
    for j in range (1,columnasA):
        Qj = Atraspuesta[j]
        for k in range (0,j):
            columnaQk = Q[:,k]
            N_ops += 2*filasA #costo de multiplicacion matricial entre 2 vectores
            #como ambos son vectores fila hago producto interno
            rkj = producto_interno(columnaQk,(Qj))
            R[k,j] = rkj
            N_ops += 2*filasA #numero filasA de multiplicaciones y restas
            Qj = Qj - rkj*columnaQk
        N_ops += 2*filasA + 1  # costo de la norma
        R[j,j] = norma2(Qj)
        #si rjj es practicamente 0 asigno 0 a la columna qj (a mayor rjj mas independiente es Qj, si es 0 es LD de los anteriores)
        if R[j,j] < tol:
            Q[:, j] = 0
            continue
        N_ops += filasA #filasA divisiones
        Q[:,j] = (Qj/R[j,j])
    
    if retornanops:
        return Q,R,N_ops
    else:
        return Q,R
#Ejemplo (da bien)
#A = np.array([[1., 2.],[3., 4.]])

#print (QR_con_GS(A, retornanops=True))


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
    m, n = A.shape
    if m < n:
        return None
    
    R = A.copy().astype(float)
    Q_full = np.identity(m).astype(float) # Inicializar Q como I_m (m x m)

    for k in range(n):
        # 1. Construcción del vector de Householder u (misma lógica que antes)
        X = R[k:m, k].copy()
        alpha = -signo(X[0]) * norma2(X)
        
        e1 = np.zeros(m - k)
        e1[0] = 1 # e1 vector canónico de R^(m-k)
        u = X - (alpha * e1)
        
        if norma2(u) > tol:
            u = u/norma2(u)
            
            # --- APLICACIÓN A R (R <- H_k R) ---
            # Bloque R[k:m, k:n]
            R_sub = R[k:m, k:n]
            
            # Calculamos la proyección: u^T * R_sub
            UporR = multiplicacionMatricialConNumpy(u, R_sub) # Resultado es 1 x (n-k)
            UporR_vector = UporR[0] # Extraemos el vector 1D
            
            # R_sub = R_sub - 2 * u @ (u^T @ R_sub)
            R[k:m, k:n] = R_sub - 2 * productoExterior(u, UporR_vector)
            
            # Fijamos los ceros y alpha en la columna k
            R[k, k] = alpha 
            R[k+1:m, k] = 0.0
            
            # --- APLICACIÓN A Q (Q <- Q H_k^T) ---
            # Aplicamos la misma reflexión H_k (simétrica) a la matriz Q_full
            
            Q_sub = Q_full[:, k:m] # Bloque Q[0:m, k:m]
            
            # Calculamos la proyección: Q_sub @ u
            Q_u = multiplicacionMatricialConNumpy(Q_sub, u) # Resultado es m x 1
            Q_u_vector = Q_u.flatten() # Extraemos el vector 1D
            
            # Q_sub = Q_sub - 2 * (Q_sub @ u) @ u^T
            # Usamos Q_u_vector (m) y u (m-k)
            # Reajustamos Q_full[0:m, k:m]
            Q_full[:, k:m] = Q_sub - 2 * productoExterior(Q_u_vector, u)

    # 2. Devolver la QR Reducida para el TP (Q m x n, R n x n)
    Q_reducida = Q_full[:, :n] # Las primeras n columnas de Q_full
    R_reducida = R[:n, :n]     # El bloque superior n x n de R
    
    return Q_reducida, R_reducida
#print(QR_con_HH(A))
metodos = ["RH","GS"]
def calculaQR (A, metodo="RH", tol=1e-12,retornanops = False):
    if metodo not in metodos:
        return None
    if metodo == "GS":
        return QR_con_GS(A,tol,retornanops)
    elif metodo == "RH":
        return QR_con_HH(A,tol)
