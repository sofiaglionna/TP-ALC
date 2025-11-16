import numpy as np

#======================================
# FUNCIONES AUXILIARES utilizadas en los módulos
#======================================

# Para importar alguna función de acá pongan en su archivo: 
# from AUXILIARES import nombre_funcion

#======================================

# Funciones auxiliares en este archivo:

#) esCuadrada(A)

#) esSimetricaConTol(A, atol=1e-10)

#) esSimetrica(A)

#) traspuesta(A)

#) abs(x)

#) transformar(A,x)

#) multiplicacionMatricial(A,B)

#) producto_interno(v, w)

#) producto_externo(v, w)

#) inversa(A)

#) extraer_sup(A)

#======================================

def esCuadrada(A):
    if A.shape[0] == A.shape[1]:
        return True
    else:
        return False


def esSimetricaConTol (A, atol=1e-10):
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            # Comprobar si A[i, j] es significativamente diferente de A[j, i] (A^t)
            # Esto se hace comparando el valor absoluto de la diferencia con atol.
            if abs(A[i, j] - A[j, i]) > atol:
                return False # No es simétrica
    return True


def esSimetrica(A):
    trasp = traspuesta(A)
    if esCuadrada(A):
        if np.array_equal(A, trasp) == True:
            return True
        else:
            return False
    else:
        return False


def traspuestaConNumpy(A):   
    res = []
    #si es un vector
    if len(A.shape) == 1:
        for i in range(0,A.shape[0]):
            res.append([A[i]])
        return np.array(res)
    
    filas, columnas = A.shape
    for i in range(0,columnas):
        columna = []
        for j in range(0,filas):
            columna.append(A[j][i])
        res.append(columna)
    return np.array(res)


#traspuesta de una matriz (devuelve una lista de listas, no una matriz)
def traspuesta(A):
  resultado = []
  for i in range(len(A)):
    #si A es un vector devuelvo un vector columna (en el otro caso se rompe si es un vector por len(A[0]))
    if type(A[0]) != list:
        for i in A:
            resultado.append([i])
        return resultado
    vector = []
    for j in range(len(A[0])):
      vector.append(A[j][i])
    resultado.append(vector)
  return resultado

#le aplica el absoluto a todos los elementos de una lista
def abs(x):
  result = []
  for elem in x:
    if elem >= 0:
      result.append(elem)
    if elem < 0:
      result.append(-elem)
  return result

#calcula la transformacion de un vector x por la matriz A
def transformar(A,x):
  vector = []

  for i in range(len(A)):
    suma = 0
    for j in range(len(A[0])):
      suma = suma + A[i][j]*x[j]
    vector.append(suma)
  return vector


def multiplicacionMatricialConNumpy (A,B):
    # Si A es un vector va a fallar .shape de numpy, por lo que lo convierto a matriz de 1 fila
    if len(A.shape) == 1:
        A = A.reshape(1, -1)
    # Lo mismo con B pero este solo puede ser un vector columna por lo que lo convierto a matriz de 1 columna
    if len(B.shape) == 1:
       B = B.reshape(-1, 1)
    if (A.shape[1] != B.shape[0]):
        return "No se puede calcular, dimensión incompatible"
    
    #traspongo B para optimizar
    BT = traspuestaConNumpy(B)
    
    res=np.zeros((A.shape[0],B.shape[1]))
    
    #me guardo ya los shapes para optimizar
    NfilasA, NcolumnasA = A.shape
    NfilasBT, NcolumnasBT = BT.shape
    
    #itero en las filas de A
    for l in range(0,NfilasA):
        filaA = A[l]
        #itero en las columnas de B para una misma fila de A
        for i in range(0,NfilasBT):
            filaBT = BT[i]
            valorli = 0
            #calculo el valor de la posicion (l,i) multiplicando fila por columna
            for j in range(0, NcolumnasA):
                #la fila j de BT es la columna j de B
                valorli += filaA[j] * filaBT[j]
            res[l,i] = valorli
    return res


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

    
def inversa(A):
    n = A.shape[0] #filas de A
    m = A.shape[1] #columnas de A

    if n != m: return "La matriz no tiene inversa; no es cuadrada"

    I = np.zeros((n, n))
    for i in range(n):
        I[i, i] = 1.0

    A_ext = np.zeros((n, 2 * n))
    for i in range(n):
        for j in range(n):
            A_ext[i, j] = A[i, j]
        for j in range(n):
            A_ext[i, j + n] = I[i, j]

    for i in range(n):
        if A_ext[i, i] == 0:
            for k in range(i + 1, n):
                if A_ext[k, i] != 0:
                    A_ext[[i, k]] = A_ext[[k, i]]
                    break
            else:
                return "La matriz no tiene inversa; pivote nulo"

        pivote = A_ext[i, i]
        for j in range(2 * n):
            A_ext[i, j] /= pivote

        for k in range(n):
            if k != i:
                factor = A_ext[k, i]
                for j in range(2 * n):
                    A_ext[k, j] -= factor * A_ext[i, j]

    A_inv = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A_inv[i, j] = A_ext[i, j + n]

    return A_inv

def extraer_sup(A):
  n = A.shape[1]
  Ac = A.copy()

  for k in range(1,n):
    for i in range(k):
      Ac[k,i] = 0
  return Ac
