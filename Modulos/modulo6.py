import numpy as np
from AUXILIARES import producto_interno,absoluto, producto_externo, esSimetricaConTol, f_A, f_A_kveces,norma, traspuestaConNumpy, multiplicacionMatricialConNumpy as multiplicacionMatricial

# Funciones del Módulo

def metpot2k(A, tol=1e-15, K=1000):
    n = A.shape[0]
    v = np.random.rand(n)  # Genero un vector.
    v  = v/(norma(v,2))

    v_barra = f_A_kveces(A, v, 2)
    v_barra = v_barra / norma(v_barra, 2)
    e = producto_interno((v_barra), v)  # Medidor de parentezco entre v_barra y v.
    k = 0  # Cantidad de iteraciones.

    while absoluto(e - 1) > tol and k < K:
        v = v_barra
        v_barra = f_A(A, v)
        v_barra = v_barra / norma(v_barra, 2)
        e = producto_interno((v_barra), v)
        k += 1

    Av = multiplicacionMatricial(A, v_barra)
    landa = producto_interno((v_barra), Av)#[0]#comentom  porque da error en el tp, testear  # El autovalor.
    epsilon = absoluto(e - 1)  # El error

    return v_barra, landa, k



def diagRH(A, tol=1e-15, K=1000):
    if esSimetricaConTol(A) == False:
        return None

    v1, lambda1, _ = metpot2k(A, tol, K)

    n = A.shape[0]
    e1 = np.zeros(n)
    e1[0] = 1# e1 es el primer vector canonico
    
    
     #Cambio sentido (en caso de que e1 y v1 sean muy similares u queda casi 0. si v1 es muy similar a -e1 me quedaria
     #u muy similar a -2e1) v1 y -v1 son el mismo autovector matematicamente, tomo el v1 positivo para que sea similar
     #a e1 (correcto con algoritmo de householder, justamente busco esto, que v1 y e1 sean similares)
    if v1[0] < 0:
        v1 = -v1

    u = v1 - e1
    if norma(u, 2) < 1e-12:# tolerancia que usamos para House Holder en modulo 5
        # En este caso, Hv1 es muy parecido a la identidad y no aplicamos House Holder
        S = np.identity(n)
        D = np.zeros((n, n))
        D[0, 0] = lambda1

        if n > 1:
            S_moño, D_moño = diagRH(A[1:, 1:], tol, K)
            D[1:, 1:] = D_moño
            S[1:, 1:] = S_moño

        # ordenamos la diagonal
        Diagonal = list(np.diag(D))
        #reordeno de mayor a menor, en la posicion i esta el indice que marca donde se encuentra el elemento numero i en orden
        indices = []
        for i in range(0, len(Diagonal)):
            indice= 0
            for j in range(0,len(Diagonal)):
                if Diagonal[i]<Diagonal[j]:
                    indice += 1
            indices.append(indice)

        D_ordenada = np.zeros(D.shape)
        S_ordenada = np.zeros(S.shape)

        for i_viejo, i_nuevo in enumerate(indices):
            D_ordenada[i_nuevo, i_nuevo] = D[i_viejo, i_viejo]
            S_ordenada[:, i_nuevo] = S[:, i_viejo]

        D = D_ordenada
        S = S_ordenada

        return S, D

    Hv1 = np.identity(n) - 2 * (producto_externo(u, u) / producto_interno(u, u))
    if n == 2:
        S = Hv1
        TodaD = multiplicacionMatricial(multiplicacionMatricial(Hv1, A), traspuestaConNumpy(Hv1))
        # Generamos la diagonal a mano, sino nos pueden quedar valores de punto flotante que dan errores, hago esto solo en el caso base y en el resto se respeta por recursion
        Diagonal2 = [TodaD[0, 0], TodaD[1, 1]]
        D = np.zeros((2, 2))
        D[0, 0] = Diagonal2[0]
        D[1, 1] = Diagonal2[1]
    else:
        B = multiplicacionMatricial(multiplicacionMatricial(Hv1, A), traspuestaConNumpy(Hv1))
        A_moño = B[1:, 1:]
        #Al hacer diagRH recursivamente vamos acumulando error que podria dar que no sea simetrico (pase la tolerancia)
        #con este paso intermedio nos aseguramos que respete la simetria en cada iteracion:
        A_moño = (A_moño + traspuestaConNumpy(A_moño)) / 2
        S_moño, D_moño = diagRH(A_moño, tol=tol, K=K)

        D = np.zeros((n, n))
        D[0, 0] = lambda1
        D[1:, 1:] = D_moño

        auxiliar = np.zeros((n, n))
        auxiliar[0, 0] = 1
        auxiliar[1:, 1:] = S_moño
        S = multiplicacionMatricial(Hv1, auxiliar)

    # ordenamos la diagonal
    diag_vals = list(np.diag(D))
    #reordeno de mayor a menor, en la posicion i esta el indice que marca donde se encuentra el elemento numero i en orden
    indices = []
    for i in range(0, len(diag_vals)):
        indice= 0
        for j in range(0,len(diag_vals)):
            if diag_vals[i]<diag_vals[j]:
                indice += 1
        indices.append(indice)

    D_ordenada = np.zeros(D.shape)
    S_ordenada = np.zeros(S.shape)

    for i_viejo, i_nuevo in enumerate(indices):
        D_ordenada[i_nuevo, i_nuevo] = D[i_viejo, i_viejo]
        S_ordenada[:, i_nuevo] = S[:, i_viejo]

    D = D_ordenada
    S = S_ordenada

    return S, D
    return S, D

