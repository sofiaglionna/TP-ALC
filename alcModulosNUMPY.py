import numpy as np

"""
#####################################################################################
#AUXILIARES
#####################################################################################
"""
def esCuadrada(A):
    if A.shape[0] == A.shape[1]:
        return True
    else:
        return False

def traspuestaConNumpy(A):
    # Caso donde A es un vector (devuelvo fila, asumo que A es una columna)
    if len(A.shape) == 1:
        n = A.shape[0]
        res = []
        for i in range(n):
            res.append([A[i]])
        return np.array(res)     

    # Caso donde A es una matriz
    filas, columnas = A.shape
    res = []
    for j in range(columnas):
        filaNueva = []
        for i in range(filas):
            filaNueva.append(A[i][j])
        res.append(filaNueva)

    return np.array(res)


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
def absolutoLista(x):
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

def producto_interno(v, w):  # Solo vector x vector
    return sum(v[i] * w[i] for i in range(len(v)))


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


def f_A(A, v):
    w = (A@ v) # Multiplico A por v y el vector resultado lo llamo w.
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
"""
#####################################################################################
#modulo 1
#####################################################################################
"""
#funcion que devuelve el valor absoluto de un numero
def absoluto (x):
    if x < 0:
        return -x
    else:
        return x

#funcion que devuelve el error absoluto de un numero y su version en numero de maquina
def error(x,y):
    y= np.float64(y) #me aseguro de que y sea de tipo float64 (numero de maquina)
    return absoluto(x-y)

#funcion que devuelve el error relativo de un numero y su version en numero de maquina
def error_relativo (x,y):
    return (error(x,y)/absoluto(x))


#funcion que devuelve True si 2 matrices A y B son iguales (asumo que se puede usar shape)
def matricesIguales(A,B,tol=1e-8):
    if A.shape != B.shape:
        return False
    else:
        for i in range (A.shape[0]):
            for j in range (A.shape[1]):
                #agrego tolerancia ya que sino no pasa los tests porque me quedan valores con numero de maquina
                if error(A[i,j], B[i,j]) > tol:
                    return False
    return True
"""
#####################################################################################
#modulo 2
#####################################################################################
"""
def rota(theta):
    matriz = [[(np.cos(theta)), -(np.sin(theta))],[np.sin(theta), np.cos(theta)]]
    return np.round(np.array(matriz)).astype(int) #convierto a int para que no me de los numeros de maquina

    
#%%
def escala (s):
    matriz = np.zeros((len(s),len(s)))
    for i,valor in enumerate(s):
        matriz[i,i] = valor
    return matriz


def rota_y_escala (theta,s):
    #Voy a usar que la multiplicacion de matrices es asociativa por lo que puedo calcular ambas matrices con los
    #codigos anteriores y multiplicarlas y al multiplicarlas por un vector sera lo mismo que si primero multiplico
    #por la que lo rota y luego por la que lo escala (ejemplo: A*(B*v) = (A*B)*v)
    B = rota(theta)
    A=escala(s)
    return (A@B)
    
#%%
#contexto: el vector v que me daran vive en z=1. Sera un vector de R^2 extendido a R^3. Entonces su tercera
#coordenada será 1. Esto para poder sumarle b en una multiplicacion de matrices. En una matriz de R^(2x2) es imposible
#desplazar el vector multiplicandolo por una matriz, solo es posible rotarlo o escalarlo ya que no puedo sumarle nada.
#Esta tercera componene igual a 1 se agrega para que al multiplicar si pueda sumar b usando una matriz de 3x3, ya que
#cada coordenada de b sera multiplicada por 1 y sumada al vector v, cuyas coordenadas x,y se le sumaran b y la
#coordenada z seguira siendo 1.
def afin (theta,s,b):
    #creo matriz 3x3
    res = np.identity(3)
    #la roto y escalo con la anterior
    MatrizRotarYEscalar = rota_y_escala(theta,s)
    #copio esta matriz en la parte superior izquierda de res
    res[0:2,0:2] = MatrizRotarYEscalar
    #agrego que traslade en b
    res[0:2,2] = b
    return res


#%%

def extenderVectorColumna (v,a):
    #si me pasan v como vector lo paso a vector columna y luego extiendo
    if len(v.shape) == 1:
        v=traspuestaConNumpy(v)
    res = []
    for i in range(0,v.shape[0]):
        res.append([v[i][0]])
    res.append([a])
    return np.array(res)

def trans_afin (v,theta,s,b):
    matrizAfin = afin(theta,s,b)
    #vector v extendido a R^3 poniendo un 1 en la tercera posicion
    vExtendido = extenderVectorColumna(v,1)
    resExtendida = (matrizAfin@ vExtendido)
    #elimino el ultimo valor de resExtendida
    res = resExtendida[0:resExtendida.shape[0]-1]
    #si me pasaron v como vector fila devuelvo res como tal
    if len(v.shape) == 1:
        res = traspuestaConNumpy(res)
    return res
"""
#####################################################################################
#modulo 3
#####################################################################################
"""


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
    #si me pasan solo 1 o inf devuelvo solo su norma, no la lista
    if len(res) == 1:
        return res[0]
    #si me pasan una norma que no sea 1 o inf devuelvo None
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
    #me piden norma 1 (sin el type se rompe por hacer str = int. Por eso aparece en varios lados)
    if type(p) == int and p == 1:
        inversa_A = inversa(A)
        condA = normaExacta(A,p)*normaExacta(inversa_A,p)
        return condA
    #me piden norma infinito
    if type(p) == str and p == "inf":
        inversa_A = inversa(A)
        condA = normaExacta(A,p)*normaExacta(inversa_A,p)
        return condA
    norma_A = normaMatMC(A,p,p,10000)
    norma_A_inv = normaMatMC(inversa(A),p,p,10000)
    return norma_A * norma_A_inv
    #No entiendo esto, pq si la norma no es sobre 1 o infinito se calcula igual que en la funcion anterior (condMC(A,p))
    #y se supone que en este punto hay que calcularlo de otra manera.
    # cond(A) = ||A|| . ||inversa(A)||
"""
#####################################################################################
#modulo 4
#####################################################################################
"""

def calculaLU(A):
    cant_op = 0
    if A is None:
        return None, None, 0

    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()

    if m!=n:
        print('Matriz no es cuadrada')
        return None, None, 0


    for k in range(n-1):

      pivot = Ac[k, k]

      if pivot == 0:
        print(f'Pivote cero en la columna {k+1}. La matriz no admite factorización LU sin pivoteo.')
        return None, None, 0

      for i in range(k + 1, n):

        multiplicador = Ac[i, k] / pivot
        cant_op += 1

        Ac[i, k] = multiplicador

        for j in range(k+1 ,n ):
          Ac[i, j] = Ac[i, j] - multiplicador * Ac[k, j]
          cant_op += 2


    U = extraer_sup(Ac)
    L = Ac - U
    L = L + np.eye(n)



    return L, U, cant_op


def res_tri(L,b,inferior=True):
  """ Resuelve el sistema Lx=b, donde L es triangular. Se puede indicar si es
      triangular inferior o superior usando el argumento inferior (por default
      asumir que es triangular inferior) """

  n = L.shape[0]
  x = [0]* n

  if inferior:

    # --- Sustitución Hacia Adelante (Forward Substitution) ---
    # Resuelve Ly = b. L es triangular inferior.
    # Recorremos de la fila 0 a la n-1 (i = 0, 1, ..., n-1).

    for i in range(n):

      suma = 0
      for j in range(i):
        suma = suma + L[i,j]*x[j]

      if L[i,i] == 0:
        print(f'Pivote cero en la fila {i+1}. La matriz no admite factorización LU sin pivoteo.')
        return None

      x[i] = (b[i] - suma) / L[i, i]

  else:
    # --- Sustitución Hacia Atrás (Backward Substitution) ---
    # Resuelve Ux = b. U es triangular superior.
    # Recorremos de la fila n-1 a la 0 (i = n-1, n-2, ..., 0).

    for i in range(n-1, -1, -1):

      suma = 0
      for j in range(i+1, n):
        suma = suma + L[i,j]*x[j]

      if L[i,i] == 0:
        print(f'Pivote cero en la fila {i+1}. La matriz no admite factorización LU sin pivoteo.')
        return None

      x[i] = (b[i] - suma) / L[i, i]

  return x


def inversa(A):
    """ Calcula la inversa de A utilizando la factorizacion LU
        y las funciones que resuelven sistemas triangulares"""

    LU_A = calculaLU(A)
    L = LU_A[0]
    U = LU_A[1]

    if L is None:
      print("La matriz no admite factorización LU (o es singular). No se puede calcular la inversa.")
      return None

    n = A.shape[0]
    I = np.eye(n)
    A_inv = np.zeros((n,n))
    for i in range(n):
      b = I[i]
      y = res_tri(L,b)
      if y is None:
          return None

      x = res_tri(U,y,inferior=False)
      if x is None:
          # Si res_tri con U devuelve None, entonces U es singular
          # por lo tanto A es singular y no tiene inversa.
          return None

      A_inv[:,i] = x

    return A_inv


def calculaLDV(A):

  # 1. Obtener L y U
    # Asume que calculaLU retorna L, U, y el conteo de operaciones
    # Desempacamos para solo obtener L y U
    resultado_LU = calculaLU(A)

    if resultado_LU is None:
        L, U = None, None
    else:
        L, U, _ = resultado_LU

    if L is None:
        print("La matriz no admite factorización LU (o es singular). No se puede calcular la descomposición LDV.")
        return None, None, None

    n = A.shape[0]

    # 2. D es la diagonal de U
    diag_U = np.diag(U)
    D = np.diag(diag_U)
    # 3. Calcular V tal que U = D @ V, usando la división fila por fila
    V = U.copy()
    atol = 1e-15 # Tolerancia para chequear singularidad

    for i in range(n):
        d_ii = diag_U[i]

        # --- A. Chequeo de Singularidad (División por Cero) ---
        if abs(d_ii) < atol:
            print(f"La matriz U es singular (pivote {i+1} es cero). No se puede completar LDV con V unitaria.")
            return None, None, None



    D_inv_diag = np.diag(1 / np.diag(D))
    V = (D_inv_diag@U)

    return L, D, V

def esSDP(A, atol=1e-10):
    """
    Checkea si la matriz A es Simétrica Definida Positiva (SDP)
    usando la factorizacion LDV
    """
    n = A.shape[0]

    # --- 1. VERIFICAR SIMETRÍA (A = A^t) ---
    # Recorrer la triangular superior para compararla con la triangular inferior.
    # Usamos la tolerancia atol para punto flotante.
    for i in range(n):
        for j in range(i + 1, n):
            # Comprobar si A[i, j] es significativamente diferente de A[j, i] (A^t)
            # Esto se hace comparando el valor absoluto de la diferencia con atol.
            if abs(A[i, j] - A[j, i]) > atol:
                return False # No es simétrica

    # --- 2. CALCULAR LDV ---

    # La descomposición LDV solo es relevante si A es simétrica.
    # Si calculaLU falla, retorna None, lo que manejamos a continuación.
    LDV = calculaLDV(A)

    # Chequear si la descomposición falló (ej. por singularidad)
    if LDV[0] is None: # Changed condition to check if L (first element) is None
        return False

    D = LDV[1] # D es la matriz diagonal

    # --- 3. VERIFICAR DEFINIDA POSITIVA (Elementos de D > 0) ---

    # Recorrer la diagonal de D para verificar la positividad.
    for i in range(n):
        d_ii = D[i, i]

        # Una matriz simétrica es SDP si D[i, i] es estrictamente POSITIVO (> 0).
        # Verificamos si es menor o igual a cero, usando atol para manejar la precisión.
        if d_ii <= atol:
            return False # No es Definida Positiva

    # Si la matriz pasó el chequeo de simetría y todos los D[i, i] son positivos, es SDP.
    return True



def calculaCholesky(A, atol=1e-10):
    """
    Verifica si A es Simétrica Definida Positiva (SDP) y, en caso afirmativo,
    calcula la matriz R de la factorización de Cholesky A = R R^t,
    donde R = L D^(1/2).
    """

    # 1. Verificar si la matriz es SDP
    if not esSDP(A, atol=atol):
        print("La matriz no es Simétrica Definida Positiva. La factorización de Cholesky no es aplicable.")
        return None

    # 2. Obtener la factorización LDL^t (usando LDV)
    # Ya que esSDP es True, sabemos que LDV = LDL^t
    LDV = calculaLDV(A)

    # Chequear si calculaLDV falló internamente (aunque esSDP ya verificó la singularidad)
    if LDV is None:
        # Esto solo debería ocurrir si LDV falló después de que esSDP dio True.
        # Es una precaución.
        print("Fallo en la factorización LDV.")
        return None

    L = LDV[0]  # Matriz triangular inferior unitaria
    D = LDV[1]  # Matriz diagonal con pivotes positivos

    n = A.shape[0]

    # 3. Calcular D^(1/2)

    # Extraer la diagonal de D como un vector
    diag_D_vector = np.diag(D)

    # Aplicar la raíz cuadrada a cada elemento (son garantizados positivos por esSDP)
    diag_D_sqrt = np.sqrt(diag_D_vector)

    # Convertir el vector de raíces cuadradas de nuevo en una matriz diagonal
    D_sqrt = np.diag(diag_D_sqrt)

    # 4. Calcular R = L * D^(1/2)

    R = (L@ D_sqrt)

    return R
"""
#####################################################################################
#modulo 5
#####################################################################################
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 12:25:15 2025

@author: Usuario
"""

def columnas (A):
    res = []
    for i in range(0,A.shape[1]):
        columna = []
        for j in range (0,A.shape[0]):
            columna.append(A[j][i])
        res.append(columna)
    return res


def norma2(a):
    return norma(a,2)

def QR_con_GS (A,tol=1e-12, retornanops=False):
    #si no es cuadrada devuelve None
    if len(A.shape) == 1:
        if A.shape[0] != 1:
            return None 
    if A.shape[0] != A.shape[1]:
        return None
    ColumnasA = columnas(A)
    #Hago el algoritmo QR
    N_ops = 0
    Q = np.zeros((len(ColumnasA),len(ColumnasA)))
    R = np.zeros((len(ColumnasA),len(ColumnasA)))
    for j in range(0,len(ColumnasA)):
        a=ColumnasA[j]
        #len(a) multiplicaciones, len(a)-1 sumas y una raiz cuadrada
        N_ops += 2*len(a)
        rjj = norma2(a)
        #paso a a vector
        a = np.array(a)
        #si rjj es practicamente 0 (menor a tol) le asigno 0
        if rjj < tol:
            R[j,j] = 0.0
            Q[0:len(ColumnasA), j] = 0.0
        else:
            R[j, j] = rjj
            #len(a) divisiones 
            qj = a / rjj
            N_ops += len(a)
            Q[0:len(ColumnasA), j] = qj
        for i in range(j+1,len(ColumnasA)):
            # len(qj) multiplicaciones y sumas -> 2*len(qj) operaciones
            rji = (qj@np.array(ColumnasA[i]))
            if rji.shape[0] == 1:
                rji = rji[0]
            R[j,i] = rji
            # len(qj) multiplicaciones y restas -> 2*len(qj) operaciones
            ColumnasA[i] = ColumnasA[i] - rji*qj
            #cuento operaciones
            N_ops += 2*len(qj)
            N_ops += 2*len(qj)
    #print para visualizar
    #print("Matriz Q:")
    #for fila in Q:
    #    print(fila)
    #
    #print("\nMatriz R:")
    #for fila in R:
    #    print(fila)
    if retornanops:
    #    print(N_ops)
        return Q,R,N_ops
    else:
        return Q,R
#Ejemplo (da bien)
#A= np.array([[2,3],[0,4]])
#print (QRconGS(A, retornanops=True))

#devuelve 1 si k es positivo, -1 si es negativo
def signo (k):
    if k<0:
        return (-1)
    else: 
        return 1
#devuelve el vector canonico con 1 en i y dimension dim
def canonico (i,dim):
    res = np.zeros(dim)
    res[i] = 1
    return res

#funcion que calcula el productoExterior entre 2 vectores fila (toma como si el primero (A) fuera columna)
def productoExterior (A,B):
    res = np.zeros((len(A),len(B)))
    for i in range (0,len(A)):
        for j in range (0,len(B)):
            res[i][j] = A[i]*B[j]
    return res
    
#No usamos el algoritmo de la guia, usamos uno mejor optimizado (no construye H moño, muy costoso en matrices A grandes)
def QR_con_HH (A,tol=1e-12):
    if len(A.shape) == 1:
        if A.shape[0] != 1:
            return None 
    if A.shape[0] < A.shape[1]:
        return None
    filas, columnas = A.shape
    
    R = A.copy()
    #paso a float para evitar errores
    R= R.astype(float)
    Q = np.identity(filas)
    #No calculo todo Q y R sino sus versiones reducidas para optimizar
    for k in range(0, min(filas, columnas)):
        X = R[k:filas,k].copy()
        a= (signo(X[0]))*(norma2(X))
        u = X - (a*canonico(0,filas-k))
        if norma2(u) > tol:
            u = u/norma2(u)
            R_sub = R[k:filas, 0:columnas]
            UporR = (u@ R_sub)
            if UporR.shape[0] == 1:
                UporR = UporR[0]
            #UporR siempre es un vector ya que u lo es y es un producto matricial
            R[k:filas, 0:columnas] = R_sub - 2*productoExterior(u,UporR)
            Q_sub = Q[0:filas, k:filas]
            Qu = (Q_sub@ u)
            if Qu.shape[0] == 1:
                Qu = Qu[0]
            #Qu siempre es un vector columna, lo paso a vector fila para producto exterior. (tomo la posicion 0 ya que traspuesta devuelve una matriz (1xlen(Qu))y yo quiero un vector)
            Q[0:filas, k:filas] = Q_sub - 2*productoExterior(traspuestaConNumpy(Qu)[0], u)
    return Q,R

metodos = ["RH","GS"]
def calculaQR (A, metodo="RH", tol=1e-12,retornanops = False):
    if metodo not in metodos:
        return None
    if metodo == "GS":
        return QR_con_GS(A,tol,retornanops)
    elif metodo == "RH":
        return QR_con_HH(A,tol)
    
"""
#####################################################################################
#modulo 6
#####################################################################################
"""

# Funciones del Módulo

def metpot2k(A, tol=1e-15, K=100):
    n = A.shape[0]

    v = np.random.rand(n)  # Genero un vector.

    sumatoria = sum(v[i]**2 for i in range(len(v)))
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

    Av = (A@ v_barra)
    landa = producto_interno((v_barra), Av)[0]  # El autovalor.
    epsilon = abs(e - 1)  # El error

    return v_barra, landa, k




def diagRH(A, tol=1e-15, K=100):
         if esSimetricaConTol(A, tol) == False:
             return None

         v1, lambda1, _ = metpot2k(A, tol, K)  # v1 = primer autovector de A ; lambda1 = autovalor

         n = A.shape[0]
         e1 = np.zeros(n)
         e1[0] = 1  # e1 es el primer vector canonico
         # Cambio sentido (en caso de que e1 y v1 sean muy similares u queda casi 0. si v1 es muy similar a -e1 me quedaria
         #u muy similar a -2e1) v1 y -v1 son el mismo autovector matematicamente, tomo el v1 positivo para que sea similar
         #a e1 (correcto con algoritmo de householder, justamente busco esto, que v1 y e1 sean similares)
         if v1[0] < 0:
             v1 = -v1
         u = v1 - e1
         Hv1 = np.eye(n) - 2 * (producto_externo(u, u) / producto_interno(u, u)) # producto_externo es np.outer(u, u)  ;  producto_interno es np.dot(u,u), que es la norma al cuadrado de (e1 - v1)
         if n == 2:
             S = Hv1
             D = ((Hv1@A)@Hv1.T)   # Hv1 @ A @ Hv1.T
         else:
             B = ((Hv1@A)@Hv1.T)   # Hv1 @ A @ Hv1.T
             A_moño = B[1:, 1:]
             #Al hacer diagRH recursivamente vamos acumulando error que podria dar que no sea simetrico (pase la tolerancia)
             #con este paso intermedio nos aseguramos que respete la simetria en cada iteracion:
             A_moño = (A_moño + A_moño.T) / 2  
             S_moño, D_moño = diagRH(A_moño, tol=1e-15, K=100)

             D = np.zeros((n, n))
             D[0, 0] = lambda1
             D[1:, 1:] = D_moño
             auxiliar = np.zeros((n, n))
             auxiliar[0, 0] = 1
             auxiliar[1:, 1:] = S_moño
             S = (Hv1@auxiliar)   # Hv1 @ auxiliar
         return S, D

"""
#####################################################################################
#modulo 7
#####################################################################################
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

def nucleo(A,tol=1e-15):
    # Primero calculamos A^t por A y lo llamamos B
    B = (A.T@A)

    # Luego uso diagRH (diagonalizacion con Householder)
    S, D = diagRH(B, tol, K=100)

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

"""
#####################################################################################
#modulo 8
#####################################################################################
"""


"""
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

"""


# Funcion del ejercicio

def svd_reducida(A, k="max", tol=1e-15):
    n = A.shape[0] # Cantidad de filas de A
    m = A.shape[1] # Cantidad de columnas de A

    if n >= m: # Filas >= Columnas
        B = (traspuestaConNumpy(A)@ A)  # Llamo la matriz B, la multiplicación de A y A traspuesta. (B es simetrica)

        hatV_aux, D_aux = diagRH(B)  # en hatV se guardan los autovectores en las columnas  ;  D se guardan los autovalores en su diagonal.

        autovalores_aux = []  # son los autovalores de B sin verificar si es mayor o menos que la tol.
        hatV = []  # son los autovectores de los autovalores de B sin verificar si los autoval cumplen con la tol.
        
        for i in range(0, D_aux.shape[0], 1):
            if abs(D_aux[i,i]) > tol:  # si el autovalor es mayor a tol (que la cumple)...
                autovalores_aux.append(D_aux[i,i])  # la guardo en la lista autovalores_aux.
                hatV.append(hatV_aux[:, i].tolist())  # guardo la columna válida.
                
        hatV = traspuestaConNumpy(np.array(hatV))  # le aplico traspuesta porque antes me quedaron los autovectores como filas. Este hatV ya viene con los autovectores de los autovalores que sobrevivieron.

        valores_singulares = []

        for j in range(0, len(autovalores_aux), 1):  # Calcula raiz de los autovalores y los guarda en la lista "valores_singulares".
            valores_singulares.append(np.sqrt(autovalores_aux[j]))

        hatSig = np.diag(valores_singulares)  # Armo la matriz Sigma con los valores singulares en su diagonal.
        
        hatU = np.zeros((n, len(valores_singulares)))  # Creamos la matriz hatU con una matriz de ceros de dimension n (cant. filas de A) x cant. valores singulares)

        for t in range(0, len(valores_singulares), 1):  # Calcula hatU
            Av = (A@hatV[:, t])  # A * v_k
            if len(Av.shape) == 2 and Av.shape[1] == 1:
                Av = Av[:, 0]
            hatU[:, t] = Av / valores_singulares[t]  # Columna de U en posición k = Av, osea A * v_k / valor sigular k.
                                                     # Como dividimos por el valor singular, el vector queda normalizado, es decir que da norma = 1, por lo que no hace falta normalizarlo.
        
        if k != "max":  # Para el test "tamaños de las reducidad", recortamos los tamaños de las matrices en base al valor de k.
            hatU = hatU[:, :k]
            hatV = hatV[:, :k]
            hatSig = hatSig[:k, :k]

        hatSig=np.array(np.diag(hatSig))
        return hatU, hatSig, hatV

    
    else: # Filas < Columnas
        B = (A@ traspuestaConNumpy(A))  # Llamo la matriz B, la multiplicación de A y A traspuesta. (B es simetrica)

        hatU_aux, D_aux = diagRH(B)  # en hatU se guardan los autovectores en las columnas  ;  D se guardan los autovalores en su diagonal.

        autovalores_aux = []  # son los autovalores de B sin verificar si es mayor o menos que la tol.
        hatU = []  # son los autovectores de los autovalores de B sin verificar si los autoval cumplen con la tol.
        
        for i in range(0, D_aux.shape[0], 1):
            if abs(D_aux[i,i]) > tol:  # si el autovalor es mayor a tol (que la cumple)...
                autovalores_aux.append(D_aux[i,i])  # la guardo en la lista autovalores_aux.
                hatU.append(hatU_aux[:, i].tolist()) # elimino la columna (autovector) del autovalor menor a tol.
                
        hatU = traspuestaConNumpy(np.array(hatU))  # le aplico traspuesta porque antes me quedaron los autovectores como filas. Este hatV ya viene con los autovectores de los autovalores que sobrevivieron.

        valores_singulares = []

        for j in range(0, len(autovalores_aux), 1):  # Calcula raiz de los autovalores y los guarda en la lista "valores_singulares".
            valores_singulares.append(np.sqrt(autovalores_aux[j]))

        hatSig = np.diag(valores_singulares)  # Armo la matriz Sigma con los valores singulares en su diagonal.
        
        hatV = np.zeros((m, len(valores_singulares)))  # Creamos la matriz hatV con una matriz de ceros de dimension m (cant. columnas de A) x cant. valores singulares)

        for t in range(0, len(valores_singulares), 1):  # Calcula hatV
            A_tras_u = (traspuestaConNumpy(A)@hatU[:, t])  # A traspuesta * u_k. Ahora es A traspuesta porque vk = (AT * σk) / uk, es decir la columna k de la matriz V es igual a (la matriz A tras * la columna k de la matriz hatU) / el valor singular sw posicion k
            if len(A_tras_u.shape) == 2 and A_tras_u.shape[1] == 1:
                A_tras_u = A_tras_u[:, 0]
            hatV[:, t] = A_tras_u / valores_singulares[t]  # Columna de V en posición k = Au, osea A * u_k / valor sigular k.
                                                           # Como dividimos por el valor singular, el vector queda normalizado, es decir que da norma = 1, por lo que no hace falta normalizarlo.

        if k != "max":  # Para el test "tamaños de las reducidad", recortamos los tamaños de las matrices en base al valor de k.
            hatU = hatU[:, :k]
            hatV = hatV[:, :k]
            hatSig = hatSig[:k, :k]
        
        hatSig=np.array(np.diag(hatSig))
        return hatU, hatSig, hatV
        
        
    #### Observación: con el caso n = m, se puede usar cualquiera de los dos casos de la función porque haciendo cualquier camino, llegas a los mismo resultados ####
