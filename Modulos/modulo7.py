import numpy as np
from AUXILIARES import producto_interno, producto_externo, esSimetrica, f_A, f_A_kveces, traspuestaConNumpy as traspuesta, multiplicacionMatricialConNumpy as multiplicacionMatricial
from REVISAR_modulo6 import diagRH
# Funciones Auxiliares


# Funciones de f_A, f_A_kveces, metpot2k y diagRH viejas
"""def f_A(A, v):
    w = np.dot(A,v)  # Multiplico A por v y el vector resultado lo llamo w
    w_normalizado = np.linalg.norm(w, 2) # Calculo la norma dos del vector w
    res = w/w_normalizado # Normalizo el vector w

    return res



def f_A_kveces(A, v, k):
    w = v.copy()
    for i in range(0,k,1):
        w = f_A(A, w)
    
    return w



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
"""




# Funciones del Módulo

def transiciones_al_azar_continuas(n):
    A = np.random.rand(n,n)  # crea una matriz A de n x n con elementos reales entre 0 y 1

    for i in range(0,n,1):
        suma_columnas = 0
        for j in range(0,n,1):  # suma los elementos de la columna i, con j cambiando las filas
            suma_columnas += A[j][i]
        for m in range(0,n,1):  # para cada elemento de la columna i, lo divide con el valor de la suma de los elems de la misma columna
            A[m][i] = A[m][i] / suma_columnas

    return A


def transiciones_al_azar_uniformes(n,thres):
    A = np.random.rand(n,n)  # crea una matriz A de n x n con elementos reales entre 0 y 1)
    for i in range(0,n,1):  # fijación de los elems de A dependiendo de la condicion
        for j in range(0,n,1):
            if A[j][i] < thres:
                A[j][i] = 1
            else:
                A[j][i] = 0
    for i in range(0,n,1):
        suma_columnas = 0
        for j in range(0,n,1):  # suma los elementos de la columna i, con j cambiando las filas
            suma_columnas += A[j][i]
        #para tests: si no hay unos devuelvo 1 dividido la cantidad de elementos de la columna (parece ser lo que se espera en los test). Preguntar
        if suma_columnas == 0:
            A[:,i] = 1/n
        else:
            A[:,i] = (A[:, i]/suma_columnas)

    return A

##############################################################################################################
######## HAY QUE CONTEMPLAR EL CASO DE SI "SUMA_COLUMNAS" == 0 ??? (osea una columna con todos ceros) ########
##############################################################################################################


def nucleo(A,tol=1e-15):
    # Primero calculamos A^t por A y lo llamamos B
    B = multiplicacionMatricial(A.T,A)

    # Luego uso diagRH (diagonalizacion con Householder)
    S, D = diagRH(B, tol, K=1000)

    # Los autovalores de B (A^t por A) son:
    autovalores = np.diag(D)

    # Creo una lista vacias que se guardaran los vectores del Nucleo normalizados:
    v_nucleo = []

    for i in range(0, len(autovalores), 1):
        if abs(autovalores[i]) < tol:
            v = S[:,i]
            sumatoria = 0
            for j in range(0,len(v),1):
                sumatoria += (v[j])**2
            norma = np.sqrt(sumatoria)
            v = v / norma
            v_nucleo.append(v)

    if len(v_nucleo) == 0:
        return np.array([])
    else:
        # Devuelvo los vectores como columnas
        return np.array(v_nucleo).T


def crea_rala(listado, m_filas, n_columnas, tol = 1e-15):
    #si me pasan listado vacio devuelvo diccionario vacio y m_filas y n_columnas
    if listado == []:
        return {}, (m_filas, n_columnas)
    
    filas = listado[0]
    columnas = listado[1]
    valores = listado[2]

    A_dict_res = {}

    for i in range(0, len(valores), 1):
        if abs(valores[i]) < tol:
            valores[i] = 0

    for j in range(0, len(valores), 1):
        if valores[j] == 0:
            pass
        else:
            A_dict_res[(filas[j],columnas[j])] = valores[j]
            
    return A_dict_res, (m_filas, n_columnas)


def multiplica_rala_vector(A, v):
    # A es una matriz rala representada como [diccionario, (m, n)].
    A_dict_res, (m, n) = A
    
    w = np.zeros(m)  # w = [0, 0,...,0] m (cantidad de filas de A) veces 

    for clave in A_dict_res:        # "clave" es la tupla de las claves (fila, columna).
        fila = clave[0]             # primer elemento de la tupla en las claves del diccionario.
        columna = clave[1]          # segundo elemento de la tupla en las claves del diccionario.
        valor = A_dict_res[clave]   # valor en esa posición.

        w[fila] += valor * v[columna]   # la fila de w es igual al valor de la clave por la columna del vector v

    return w


#A = crea_rala(listado, 6, 4)  # ---> [{(0, 1): 10, (2, 3): 4, (5, 0): 7}, (6, 4)]   --->  np.array([0, 10, 0, 0],   # fila 0
                              #                                                                    [0, 0, 0, 0],    # fila 1
                              #                                                                    [0, 0, 0, 4],    # fila 2
                              #                                                                    [0, 0, 0, 0],    # fila 3
                              #                                                                    [0, 0, 0, 0],    # fila 4
                              #                                                                    [7, 0, 0, 0])    # fila 5
 
#v = np.array([1,2,3,4])
#  print(multiplica_rala_vector(A, v))  ------>  [20.  0. 16.  0.  0.  7.]
