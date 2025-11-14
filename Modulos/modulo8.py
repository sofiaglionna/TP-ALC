import numpy as np

#Funciones Auxiliares

def producto_interno(v, w):
    res = 0
    for i in range(0, len(v), 1):
        res += (v[i] * w[i])
    return res


def producto_externo(v, w):
    n = len(v)
    m = len(w)
    res = np.zeros((n, m))
    for i in range(0, n, 1):
        for j in range(0, m, 1):
            res[i, j] = (v[i] * w[j])
    return res


def esSimetrica(A):
    trasp = traspuesta(A)
    if esCuadrada(A):
        if np.array_equal(A, trasp) == True:
            return True
        else:
            return False
    else:
        return False



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


def multiplicacionMatricial(A, B):
    # Si A es un vector va a fallar .shape de numpy, por lo que lo convierto a matriz de 1 fila
    if len(A.shape) == 1:
        A = A.reshape(1, -1)
    # Lo mismo con B pero este solo puede ser un vector columna por lo que lo convierto a matriz de 1 columna
    if len(B.shape) == 1:
       B = B.reshape(-1, 1)
    if (A.shape[1] != B.shape[0]):
        raise ValueError("Dimensiones incompatibles para la multiplicación.")
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

    if res.shape[1] == 1:
        res = res.flatten()  # flatten() aplana un arreglo 2D o multidimensional en un vector 1D. Lo agrego para que pase el test del Labo08  "test_svd_reducida_mn(A, tol=1e-15)".
        
    return res


#def multiplicacion_de_matrices_sin_numpy(A,B):
    #n = A.shape[0] # filas de A
    #m = A.shape[1] # columnas de A
    #r = B.shape[0] # filas de B
    #s = B.shape[1] # columnas de B

    #if m == n:
     #   res = np.zeros((n, s))

      #  for i in range(0, n ,1):
       #     for j in range(0, s, 1):
        #        sumatoria = 0
         #       t = 0
          #      while t < m:
           #         sumatoria += A[i, t] * B[t, j]
            #        t += 1
             #   res[i,j] = sumatoria
       # return res

    #else:
     #   raise ValueError("Las dimensiones no son compatibles para la multiplicación de matrices.")





# Funcion del ejercicio

def svd_reducida(A, k="max", tol=1e-15):
    n = A.shape[0] # Cantidad de filas de A
    m = A.shape[1] # Cantidad de columnas de A

    if n >= m: # Filas >= Columnas
        B = multiplicacionMatricial(A.T, A)  # Llamo la matriz B, la multiplicación de A y A traspuesta. (B es simetrica)

        hatV_aux, D_aux = diagRH(B, tol, k)  # en hatV se guardan los autovectores en las columnas  ;  D se guardan los autovalores en su diagonal.

        autovalores_aux = []  # son los autovalores de B sin verificar si es mayor o menos que la tol.
        hatV = []  # son los autovectores de los autovalores de B sin verificar si los autoval cumplen con la tol.
        
        for i in range(0, D_aux.shape[0], 1):
            if abs(D_aux[i,i]) > tol:  # si el autovalor es mayor a tol (que la cumple)...
                autovalores_aux.append(D_aux[i,i])  # la guardo en la lista autovalores_aux.
                hatV.append(hatV_aux[:, i].tolist())  # guardo la columna válida.
                
        hatV = np.array(hatV).T  # le aplico traspuesta porque antes me quedaron los autovectores como filas. Este hatV ya viene con los autovectores de los autovalores que sobrevivieron.

        valores_singulares = []

        for j in range(0, len(autovalores_aux), 1):  # Calcula raiz de los autovalores y los guarda en la lista "valores_singulares".
            valores_singulares.append(np.sqrt(autovalores_aux[j]))

        hatSig = np.diag(valores_singulares)  # Armo la matriz Sigma con los valores singulares en su diagonal.
        
        hatU = np.zeros((n, len(valores_singulares)))  # Creamos la matriz hatU con una matriz de ceros de dimension n (cant. filas de A) x cant. valores singulares)

        for t in range(0, len(valores_singulares), 1):  # Calcula hatU
            Av = multiplicacionMatricial(A, hatV[:, t])  # A * v_k
            hatU[:, t] = Av / valores_singulares[t]  # Columna de U en posición k = Av, osea A * v_k / valor sigular k.
                                                     # Como dividimos por el valor singular, el vector queda normalizado, es decir que da norma = 1, por lo que no hace falta normalizarlo.

        return hatU, hatSig, hatV

    
    else: # Filas < Columnas
        B = multiplicacionMatricial(A, A.T)  # Llamo la matriz B, la multiplicación de A y A traspuesta. (B es simetrica)

        hatU_aux, D_aux = diagRH(B, tol, k)  # en hatU se guardan los autovectores en las columnas  ;  D se guardan los autovalores en su diagonal.

        autovalores_aux = []  # son los autovalores de B sin verificar si es mayor o menos que la tol.
        hatU = []  # son los autovectores de los autovalores de B sin verificar si los autoval cumplen con la tol.
        
        for i in range(0, D_aux.shape[0], 1):
            if abs(D_aux[i,i]) > tol:  # si el autovalor es mayor a tol (que la cumple)...
                autovalores_aux.append(D_aux[i,i])  # la guardo en la lista autovalores_aux.
                hatU.append(hatU_aux[:, i].tolist()) # elimino la columna (autovector) del autovalor menor a tol.
                
        hatU = np.array(hatU).T  # le aplico traspuesta porque antes me quedaron los autovectores como filas. Este hatV ya viene con los autovectores de los autovalores que sobrevivieron.

        valores_singulares = []

        for j in range(0, len(autovalores_aux), 1):  # Calcula raiz de los autovalores y los guarda en la lista "valores_singulares".
            valores_singulares.append(np.sqrt(autovalores_aux[j]))

        hatSig = np.diag(valores_singulares)  # Armo la matriz Sigma con los valores singulares en su diagonal.
        
        hatV = np.zeros((m, len(valores_singulares)))  # Creamos la matriz hatV con una matriz de ceros de dimension m (cant. columnas de A) x cant. valores singulares)

        for t in range(0, len(valores_singulares), 1):  # Calcula hatV
            A_tras_u = multiplicacionMatricial(A.T, hatU[:, t])  # A traspuesta * u_k. Ahora es A traspuesta porque vk = (AT * σk) / uk, es decir la columna k de la matriz V es igual a (la matriz A tras * la columna k de la matriz hatU) / el valor singular sw posicion k
            hatV[:, t] = A_tras_u / valores_singulares[t]  # Columna de V en posición k = Au, osea A * u_k / valor sigular k.
                                                           # Como dividimos por el valor singular, el vector queda normalizado, es decir que da norma = 1, por lo que no hace falta normalizarlo.

        return hatU, hatSig, hatV
        
        
    #### Observación: con el caso n = m, se puede usar cualquiera de los dos casos de la función porque haciendo cualquier camino, llegas a los mismo resultados ####
