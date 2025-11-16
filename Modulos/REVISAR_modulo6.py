import numpy as np
from AUXILIARES import producto_interno, producto_externo, esSimetrica, traspuestaConNumpy as traspuesta, multiplicacionMatricialConNumpy as multiplicacionMatricial


# Funciones Auxiliares:

def f_A(A, v):
    w = multiplicacionMatricialConNumpy(A, v) # Multiplico A por v y el vector resultado lo llamo w.
    w = w.flatten() # Aplano el resultado porque antes era una matriz de dimension nx1.
    
    sumatoria = 0  # Lo voy a usar para sumar todos los valores del vector w.
    for i in range(0, len(w), 1):
        sumatoria += w[i] ** 2
    norma = np.sqrt(sumatoria)  # Calculo la norma con la sumatoria.

    if norma == 0:
        return np.zeros_like(w)

    for j in range(0, len(w), 1):  # Normalizo el vector w.
        w[j] = w[j] / norma

    return w


def f_A_kveces(A, v, k):
    w = v.copy()
    for i in range(0,k,1):
        w = f_A(A, w)
    
    return w


# Funciones del Módulo

def metpot2k(A, tol=1e-15, K=1000):
    n = A.shape[0]
    
    v = np.random.rand(n)  # genero un autovector
    v_barra = f_A_kveces(A, v, 2)
    e = np.dot((v_barra.T), v)  # medidor de parentezco entre v_barra_traspuesta y v
    k = 0  # cantidad de iteraciones

    while abs(e - 1) > tol and k < K:
        v = v_barra
        v_barra = f_A(A, v)
        e = np.dot((v_barra.T), v)
        k += 1

    Av = np.dot(A, v_barra)
    landa = np.dot((v_barra.T), Av)  # el autovalor
    epsilon = abs(e - 1)  # el error

    return v_barra, landa, k,epsilon




def diagRH(A,tol=1e-15,K=1000):
    if esSimetrica(A) == False:
        return None
        
    v1, lambda1, _, _ = metpot2k(A, tol, K)  # v1 = primer autovector de A ; lambda1 = autovalor
    
    n = A.shape[0]
    e1 = np.zeros(n)
    e1[0] = 1  # e1 es el primer vector canonico

    u = e1 - v1
    Hv1 = np.eye(n) - 2 * (producto_externo(u, u) / producto_interno(u, u)) # producto_externo es np.outer(u, u)  ;  producto_interno es np.dot(u,u), que es la norma al cuadrado de (e1 - v1)

    if n == 2:
        S = Hv1
        D = multiplicacionMatricial(multiplicacionMatricial(Hv1,A),Hv1.T)   # Hv1 @ A @ Hv1.T
    else:
        B = multiplicacionMatricial(multiplicacionMatricial(Hv1,A),Hv1.T)   # Hv1 @ A @ Hv1.T
        A_moño = B[1:, 1:]
        S_moño, D_moño = diagRH(A_moño,tol=1e-15,K=1000)
        
        D = np.zeros((n, n))
        D[0, 0] = lambda1
        D[1:, 1:] = D_moño

        auxiliar = np.zeros((n, n))
        auxiliar[0, 0] = 1
        auxiliar[1:, 1:] = S_moño
        S = multiplicacionMatricial(Hv1, auxiliar)   # Hv1 @ auxiliar

    return S, D
