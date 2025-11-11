import numpy as np

#Funciones Auxiliares

def diagRH(A,tol=1e-15,K=1000):  ############################ FALTA MODIFICAR diagRH EN DONDE SE "@" POR LA FUNCION "multiplicacion_De_matrices_sin_numpy" ############################
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


def multiplicacion_de_matrices_sin_numpy(A,B):
    n = A.shape[0] # filas de A
    m = A.shape[1] # columnas de A
    r = B.shape[0] # filas de B
    s = B.shape[1] # columnas de B

    if m == n:
        res = np.zeros((n, s))

        for i in range(0, n ,1):
            for j in range(0, s, 1):
                sumatoria = 0
                t = 0
                while t < m:
                    sumatoria += A[i, t] * B[t, j]
                    t += 1
                res[i,j] = sumatoria
        return res

    else:
        raise ValueError("Las dimensiones no son compatibles para la multiplicación de matrices.")








# Funcion del ejercicio

def svd_reducida(A, k="max", tol=1e-15):
    n = A.shape[0] # Cantidad de filas de A
    m = A.shape[1] # Cantidad de columnas de A

    if n >= m: # Filas >= Columnas
        B = multiplicacion_de_matrices_sin_numpy(A.T, A)  # Llamo la matriz B, la multiplicación de A y A traspuesta. (B es simetrica)

        hatV, D = diagRH(B, tol, K)  # en hatV se guardan los autovectores en las columnas  ;  D se guardan los autovalores en su diagonal.

        

        for i in range(D.shape[0] - 1, -1, -1):  # Lo hacemos de atras para adelante para que no se desincronize las filas y columnas cuando las eliminamos.
            if abs(D[i, i]) < tol:
                D = np.delete(D, i, axis=0)  # np.delete() devuelve una nueva matriz, osea que no modifica la original. Por eso el D = np.delete().
                D = np.delete(D, i, axis=1)  # axis = 0 aplica a la fila  ;  axis = 1 aplica a la columna.
                hatV = np.delete(hatV, i, axis=1)

        autovalores = np.diag(D) # Matriz con los autovalores (ya filtrados por la tolerancia) en su diagonal.
        valores_singulares = []

        for j in range(0, len(autovalores), 1):  # Calcula raiz de los autovalores y los guarda en la lista "valores_singulares".
            valores_singulares.append(np.sqrt(autovalores[j]))

        hatSig = np.diag(valores_singulares)  # Armo la matriz Sigma con los valores singulares en su diagonal.
        
        hatU = np.zeros((n, len(valores_singulares)))  # Creamos la matriz hatU con una matriz de ceros de dimension n (cant. filas de A) x cant. valores singulares)

        for k in range(0, len(valores_singulares), 1):  # Calcula hatU
            Av = multiplicacion_de_matrices_sin_numpy(A, hatV[:, k])  # A * v_k
            hatU[:, k] = Av / valores_singulares[k]  # Columna de U en posición k = Av, osea A * v_k / valor sigular k.
                                                     # Como dividimos por el valor singular, el vector queda normalizado, es decir que da norma = 1, por lo que no hace falta normalizarlo.

        return hatU, hatSig, hatV

    
    else: # Filas < Columnas
        B = multiplicacion_de_matrices_sin_numpy(A, A.T)  # Llamo la matriz B, la multiplicación de A y A traspuesta. (B es simetrica)

        hatU, D = diagRH(B, tol, K)  # en hatU se guardan los autovectores en las columnas  ;  D se guardan los autovalores en su diagonal.

        

        for i in range(D.shape[0] - 1, -1, -1):  # Lo hacemos de atras para adelante para que no se desincronize las filas y columnas cuando las eliminamos.
            if abs(D[i, i]) < tol:
                D = np.delete(D, i, axis=0)  # np.delete() devuelve una nueva matriz, osea que no modifica la original. Por eso el D = np.delete().
                D = np.delete(D, i, axis=1)  # axis = 0 aplica a la fila  ;  axis = 1 aplica a la columna.
                hatU = np.delete(hatU, i, axis=1)

        autovalores = np.diag(D) # Matriz con los autovalores (ya filtrados por la tolerancia) en su diagonal.
        valores_singulares = []

        for j in range(0, len(autovalores), 1):  # Calcula raiz de los autovalores y los guarda en la lista "valores_singulares".
            valores_singulares.append(np.sqrt(autovalores[j]))

        hatSig = np.diag(valores_singulares)  # Armo la matriz Sigma con los valores singulares en su diagonal.
        
        hatV = np.zeros((m, len(valores_singulares)))  # Creamos la matriz hatV con una matriz de ceros de dimension m (cant. columnas de A) x cant. valores singulares)

        for k in range(0, len(valores_singulares), 1):  # Calcula hatU
            A_tras_u = multiplicacion_de_matrices_sin_numpy(A.T, hatU[:, k])  # A traspuesta * u_k. Ahora es A traspuesta porque vk = (AT * σk) / uk, es decir la columna k de la matriz V es igual a (la matriz A tras * la columna k de la matriz hatU) / el valor singular sw posicion k
            hatV[:, k] = A_tras_u / valores_singulares[k]  # Columna de V en posición k = Au, osea A * u_k / valor sigular k.
                                                           # Como dividimos por el valor singular, el vector queda normalizado, es decir que da norma = 1, por lo que no hace falta normalizarlo.

        return hatU, hatSig, hatV
        
        
    #### Observación: con el caso n = m, se puede usar cualquiera de los dos casos de la función porque haciendo cualquier camino, llegas a los mismo resultados ####
