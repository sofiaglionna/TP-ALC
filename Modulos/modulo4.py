import numpy as np
from AUXILIARES import multiplicacionMatricialConNumpy,extraer_sup


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