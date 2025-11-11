import numpy as np


# Funciones Auxiliares:

def f_A(A, v):
    w = np.dot(A,v)  # Multiplico A por v y el vector resultado lo llamo w
    w_normalizado = np.linalg.norm(w, 2) # Calculo la norma dos del vector w
    res = w/w_normalizado # Normalizo el vector w

    return res



def f_A_kveces(A, v, k):
    w = v.copy()
    for i in range(0,k,1):
        w = f_A(A, w)
    
    return w



def esSimetrica(A):
    trasp = traspuesta(A)
    if esCuadrada(A):
        if np.array_equal(A, trasp) == True:
            return True
        else:
            return False
    else:
        return False





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

    return v_barra, landa, k, epsilon




def diagRH(A, tol=1e-15, K=1000):
    if esSimetrica(A) == False:
        return None
        
    v1, lambda1, _, _ = metpot2k(A, tol, K)  # v1 = primer autovector de A ; lambda1 = autovalor
    
    n = A.shape[0]
    e1 = np.zeros(n)
    e1[0] = 1  # e1 es el primer vector canonico

    u = e1 - v1
    Hv1 = np.eye(n) - 2 * (np.outer(u, u) / np.dot(u,u)) # np.outer(n, n) es producto externo ; np.dot(u,u) es la norma al cuadrado de (e1 - v1)

    if n == 2:
        S = Hv1
        D = Hv1 @ A @ Hv1.T
    else:
        B = Hv1 @ A @ Hv1.T
        A_moño = B[1:, 1:]
        S_moño, D_moño = diagRH(A_moño,tol=1e-15,K=1000)
        
        D = np.zeros((n, n))
        D[0, 0] = lambda1
        D[1:, 1:] = D_moño

        auxiliar = np.zeros((n, n))
        auxiliar[0, 0] = 1
        auxiliar[1:, 1:] = S_moño
        S = Hv1 @ auxiliar

    return S, D
