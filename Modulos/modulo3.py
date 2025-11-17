import numpy as np

#Auxilaires:
from AUXILIARES import traspuestaConNumpy, transformar,multiplicacionMatricialConNumpy,inversa, absolutoLista

def norma(x,p):
    if p == 'inf':
        max = 0
        for elem in x:
            if abs(elem) > max:
                max = abs(elem)
        return max
    res = 0
    for elem in x:
        res = elem**p + res
    return res**(1/p)

#normaliza un vector
def normalizaVector(V,p):
    normaDeV = norma(V,p)
    if normaDeV != 0:
        return (V / normaDeV)
    else:
        #si el vector es nulo lo develvo
        return V

#normaliza una lista de vectores y los devuelve
def normaliza(X,p):
    res = []
    for x in X:
        res.append(normalizaVector(x,p))
    return res

def normaMatMC(A,q,p,Np):
    # Devuelve la norma ||A|| \ {q , p} y el vector x en el cual se alcanza el maximo.

    max_norm = 0
    n = len(A)
    vector_maximo = [0.0] * n
    # uso la funcion np.random.rand para crear vectores random
    for i in range(Np):
        x = np.random.rand(n)
        #normalizo el vector
        x = normalizaVector(x,p)
        # transformo el vector con la matriz
        Ax = transformar(A,x)
        norm_Ax = norma(Ax,q)
        if norm_Ax > max_norm:
            max_norm = norm_Ax
            vector_maximo = x

    return [max_norm, vector_maximo]


def normaExacta(A, p=[1,'inf']):
# Devuelve una lista con las normas 1 e infinito de una matriz A,
# usando las expresiones del enunciado 2.(c)
    res = []
    n,m = A.shape
    # Caso norma 1 de A, tengo que buscar la maxima suma de los |elementos| por columna
    if (((type(p) == list) and (1 in p)) or ((type(p) == int) and (p == 1))):
        #transpongo la matriz para poder usar el codigo de norma infinito.
        matriz = traspuestaConNumpy(A)
        max_norm1 = 0
        for i in range(0,m):
            suma = 0
            for j in range(0,n):
                suma = suma + abs(matriz[i][j])
            if suma > max_norm1:
                max_norm1 = suma
        res.append(max_norm1)
    # Caso norma infinito de A, tengo que buscar la maxima suma de los |elementos| por fila
    if (((type(p) == list) and ('inf' in p)) or ((type(p) == str) and (p == "inf"))):
        max_norminf= 0
        for i in range(0,n):
            suma = 0
            for j in range(0,m):
                suma = suma + abs(A[i][j])
            if suma > max_norminf:
                max_norminf = suma
        res.append(max_norminf)
    if len(res) == 0:
        return None
    return res

def condMC(A,p):
    # Devuelve el numero de condicion de A usando la norma inducida p.

    norma_A = normaMatMC(A,p,p,10000)[0]
    norma_A_inv = normaMatMC(inversa(A),p,p,10000)[0]
    return norma_A * norma_A_inv

def condExacta(A,p):
    #Devuelve el numero de condicion de A a partir de la formula de la ecuacion cond(A) = ||A|| . ||inversa(A)|| usando la norma p.
    if type(p) == int and p == 1:
        inversa_A = inversa(A)
        condA = normaExacta(A,p)[0]*normaExacta(inversa_A,p)[0]
        return condA
    if type(p) == str and p == "inf":
        inversa_A = inversa(A)
        #tomo el 0 porque como le pase p="inf" res va a tener solo la norma inf en la primera posicion
        condA = normaExacta(A,p)[0]*normaExacta(inversa_A,p)[0]
        return condA
    norma_A = normaMatMC(A,p,p,10000)
    norma_A_inv = normaMatMC(inversa(A),p,p,10000)
    return norma_A * norma_A_inv
    #No entiendo esto, pq si la norma no es sobre 1 o infinito se calcula igual que en la funcion anterior (condMC(A,p))
    #y se supone que en este punto hay que calcularlo de otra manera.
    # cond(A) = ||A|| . ||inversa(A)||
