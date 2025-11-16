import numpy as np
from AUXILIARES import producto_interno, producto_externo, esSimetrica, f_A, f_A_kveces, traspuestaConNumpy as traspuesta, multiplicacionMatricialConNumpy as multiplicacionMatricial


# Funciones del Módulo

def metpot2k(A, tol=1e-15, K=1000):
    n = A.shape[0]

    v = np.random.rand(n)  # Genero un vector.

    sumatoria = 0  # Lo voy a usar para sumar todos los valores del vector w.
    for i in range(0, len(v), 1):
        sumatoria += v[i] ** 2
    norma = np.sqrt(sumatoria)  # Calculo la norma con la sumatoria.
    for j in range(0, len(v), 1):  # Normalizo el vector v.
        v[j] = v[j] / norma

    v_barra = f_A_kveces(A, v, 2)
    e = producto_interno((v_barra), v)  # Medidor de parentezco entre v_barra y v.
    k = 0  # Cantidad de iteraciones.

    while abs(e - 1) > tol and k < K:
        v = v_barra
        v_barra = f_A(A, v)
        e = producto_interno((v_barra), v)
        k += 1

    Av = multiplicacionMatricial(A, v_barra)
    landa = producto_interno((v_barra), Av)[0]  # El autovalor.
    epsilon = abs(e - 1)  # El error

    return v_barra, landa, k




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
