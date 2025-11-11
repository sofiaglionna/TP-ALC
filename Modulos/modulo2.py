# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 19:03:28 2025

@author: Usuario
"""
import numpy as np
import math #lo uso para el calculo de seno y coseno, ver si esta permitido


def rota(theta):
    radianes = math.radians(theta)
    matriz = [[(math.cos(radianes)), -(math.sin(radianes))],[math.sin(radianes), math.cos(radianes)]]
    return np.round(np.array(matriz)) #redondeo para que no me de los numeros de maquina
#%%
def escala (s):
    matriz = np.zeros((len(s),len(s)))
    for i,valor in enumerate(s):
        matriz[i,i] = valor
    return matriz

#%%
#funcion que calcula la multiplicacion matricial de A*B 2 matrices dadas en ese orden
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
    return res

def rotayescala (theta,s):
    #Voy a usar que la multiplicacion de matrices es asociativa por lo que puedo calcular ambas matrices con los
    #codigos anteriores y multiplicarlas y al multiplicarlas por un vector sera lo mismo que si primero multiplico
    #por la que lo rota y luego por la que lo escala (ejemplo: A*(B*v) = (A*B)*v)
    B = rota(theta)
    A=escala(s)
    return multiplicacionMatricial(A,B)
    
#%%
#contexto: el vector v que me daran vive en z=1. Sera un vector de R^2 extendido a R^3. Entonces su tercera
#coordenada será 1. Esto para poder sumarle b en una multiplicacion de matrices. En una matriz de R^(2x2) es imposible
#desplazar el vector multiplicandolo por una matriz, solo es posible rotarlo o escalarlo ya que no puedo sumarle nada.
#Esta tercera componene igual a 1 se agrega para que al multiplicar si pueda sumar b usando una matriz de 3x3, ya que
#cada coordenada de b sera multiplicada por 1 y sumada al vector v, cuyas coordenadas x,y se le sumaran b y la
#coordenada z seguira siendo 1.
def afin (theta,s,b):
    #creo matriz 3x3
    res = np.identity(3)
    #la roto y escalo con la anterior
    MatrizRotarYEscalar = rotayescala(theta,s)
    #copio esta matriz en la parte superior izquierda de res
    res[0:2,0:2] = MatrizRotarYEscalar
    #agrego que traslade en b
    res[0:2,2] = b
    return res


#%%

def extenderVectorColumna (v,a):
    res = []
    for i in range(0,v.shape[0]):
        res.append([v[i][0]])
    res.append([a])
    return np.array(res)

def transafin (v,theta,s,b):
    matrizAfin = afin(theta,s,b)
    #vector v extendido a R^3 poniendo un 1 en la tercera posicion
    vExtendido = extenderVectorColumna(v,1)
    print(vExtendido)
    print(matrizAfin)
    return multiplicacionMatricial(matrizAfin, vExtendido)

