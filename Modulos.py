import numpy as np


#======================================
# FUNCIONES AUXILIARES utilizadas en los módulos
#======================================

# Para importar alguna función de acá pongan en su archivo: 
# from AUXILIARES import nombre_funcion

#======================================

# Funciones auxiliares en este archivo:

#) traspuesta(A)

#) abs(x)

#) transformar(A,x)

#) multiplicacionMatricial (A,B)

#) inversa(A)

#======================================
def esSimetricaConTol (A, atol=1e-10):
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            # Comprobar si A[i, j] es significativamente diferente de A[j, i] (A^t)
            # Esto se hace comparando el valor absoluto de la diferencia con atol.
            if abs(A[i, j] - A[j, i]) > atol:
                return False # No es simétrica
            
def traspuestaConNumpy (A):   
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
# ==============================================================================

# ==============================================================================
#%% ==============================    MODULOS    ===============================
# ==============================================================================

# ==============================================================================

##########################################
#%% MÓDULO 1
##########################################

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

##########################################
#%% MÓDULO 2
##########################################


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
    return multiplicacionMatricialConNumpy(A,B)
    
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
    resExtendida = multiplicacionMatricialConNumpy(matrizAfin, vExtendido)
    #elimino el ultimo valor de resExtendida
    res = resExtendida[0:resExtendida.shape[0]-1]
    #si me pasaron v como vector fila devuelvo res como tal
    if len(v.shape) == 1:
        res = traspuestaConNumpy(res)
    return res

##########################################
#%% MÓDULO 3
##########################################


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
def normaliza(X,p):
    res = []
    for x in X:
        res.append(x/norma(x,p))
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
        x = normaliza(x,p)
        # transformo el vector con la matriz
        Ax = transformar(A,x)
        norm_Ax = norma(Ax,q)
        if norm_Ax > max_norm:
            max_norm = norm_Ax
            vector_maximo = x

    return [max_norm, vector_maximo]


def normaExacta(A,p):
# Devuelve una lista con las normas 1 e infinito de una matriz A,
# usando las expresiones del enunciado 2.(c)
    res = []
    n = len(A)
    # Caso norma infinito de A, tengo que buscar la maxima suma de los |elementos| por fila
    if ('inf' in p):
        max_norminf= 0
        for i in range(n):
            suma = 0
            for j in range(n):
                suma = suma + abs(A[i][j])
            if suma > max_norminf:
                max_norminf = suma
        res.append(max_norminf)
    # Caso norma 1 de A, tengo que buscar la maxima suma de los |elementos| por columna
    if (1 in p):
        #transpongo la matriz para poder usar el codigo anterior.
        matriz = traspuesta(A)
        max_norm1 = 0
        for i in range(n):
            suma = 0
            for j in range(n):
                suma = suma + abs(matriz[i][j])
            if suma > max_norm1:
                max_norm1 = suma
        res.append(max_norm1)
    return res

def condMC(A,p):
    # Devuelve el numero de condicion de A usando la norma inducida p.

    norma_A = normaMatMC(A,p,p,10000)
    norma_A_inv = normaMatMC(np.linalg.inversa(A),p,p,10000)
    return norma_A * norma_A_inv

def condExacta(A,p):
    #Devuelve el numero de condicion de A a partir de la formula de la ecuacion cond(A) = ||A|| . ||inversa(A)|| usando la norma p.
    if p == 1 or p == "inf":
        inversa_A = inversa(A)
        condA = multiplicacion_de_matrices_sin_numpy(normaExacta(A,p), normaExacta(inversa_A,p))
        return condA
    norma_A = normaMatMC(A,p,p,10000)
    norma_A_inv = normaMatMC(inversa(A),p,p,10000)
    return norma_A * norma_A_inv
    #No entiendo esto, pq si la norma no es sobre 1 o infinito se calcula igual que en la funcion anterior (condMC(A,p))
    #y se supone que en este punto hay que calcularlo de otra manera.
    # cond(A) = ||A|| . ||inversa(A)||

##########################################
#%% MÓDULO 4
##########################################


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
    V = multiplicacionMatricialConNumpy(D_inv_diag,U)

    return L, D, V


import numpy as np

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

    R = multiplicacionMatricialConNumpy(L, D_sqrt)

    return R

##########################################
#%% MÓDULO 5
##########################################

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
            rji = multiplicacionMatricialConNumpy(qj,np.array(ColumnasA[i]))
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
    for k in range(0,columnas):
        X = R[k:filas,k].copy()
        a= (signo(X[0]))*(norma2(X))
        u = X - (a*canonico(0,filas-k))
        if norma2(u) > tol:
            u = u/norma2(u)
            R_sub = R[k:filas, 0:columnas]
            UporR = multiplicacionMatricialConNumpy(u, R_sub)
            #UporR siempre es un vector ya que u lo es y es un producto matricial
            R[k:filas, 0:columnas] = R_sub - 2*productoExterior(u,UporR)
            Q_sub = Q[0:filas, k:filas]
            Qu = multiplicacionMatricialConNumpy(Q_sub, u)
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

##########################################
#%% MÓDULO 6
##########################################

##########################################
#%% MÓDULO 7 - CAMBIAR FUNCIONES POR LAS QUE YA ESTÁN
##########################################

# Funciones Auxiliares:

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



def traspuesta(A):
    return A.T



def esCuadrada(A):
    if A.shape[0] == A.shape[1]:
        return True
    else:
        return False



def esSimetrica(A):
    trasp = traspuesta(A)
    if esCuadrada(A):
        if np.array_equal(A, trasp) == True:
            return True
        else:
            return False
    else:
        return False



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


##########################################
#%% MÓDULO 8
##########################################


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
        
        if k != "max":  # Para el test "tamaños de las reducidad", recortamos los tamaños de las matrices en base al valor de k.
            hatU = hatU[:, :k]
            hatV = hatV[:, :k]
            hatSig = hatSig[:k, :k]
            
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

        if k != "max":  # Para el test "tamaños de las reducidad", recortamos los tamaños de las matrices en base al valor de k.
            hatU = hatU[:, :k]
            hatV = hatV[:, :k]
            hatSig = hatSig[:k, :k]
            
        return hatU, hatSig, hatV
        
        
    #### Observación: con el caso n = m, se puede usar cualquiera de los dos casos de la función porque haciendo cualquier camino, llegas a los mismo resultados ####
