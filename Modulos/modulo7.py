import numpy as np
from AUXILIARES import absoluto,producto_interno, producto_externo, f_A, f_A_kveces, traspuestaConNumpy, multiplicacionMatricialConNumpy
from modulo6 import diagRH

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
    B = multiplicacionMatricialConNumpy(traspuestaConNumpy(A),A)

    # Luego uso diagRH (diagonalizacion con Householder)
    S, D = diagRH(B, tol, K=1000)

    # Los autovalores de B (A^t por A) son:
    autovalores = np.diag(D)

    # Creo una lista vacias que se guardaran los vectores del Nucleo normalizados:
    v_nucleo = []

    for i in range(0, len(autovalores), 1):
        if absoluto(autovalores[i]) < tol:
            v = S[:,i]
            sumatoria = 0
            for j in range(0,len(v),1):
                sumatoria += (v[j])**2
            norma2 = sumatoria**(1/2)
            v = v / norma2
            v_nucleo.append(v)

    if len(v_nucleo) == 0:
        return np.array([])
    else:
        # Devuelvo los vectores como columnas
        return traspuestaConNumpy(np.array(v_nucleo))


def crea_rala(listado, m_filas, n_columnas, tol = 1e-15):
    #si me pasan listado vacio devuelvo diccionario vacio y m_filas y n_columnas
    if listado == []:
        return {}, (m_filas, n_columnas)
    
    filas = listado[0]
    columnas = listado[1]
    valores = listado[2]

    A_dict_res = {}

    for i in range(0, len(valores), 1):
        if absoluto(valores[i]) < tol:
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

