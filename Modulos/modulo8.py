import numpy as np
from AUXILIARES import absoluto,traspuestaConNumpy, multiplicacionMatricialConNumpy,esCuadrada,f_A,f_A_kveces,producto_interno,producto_externo
from modulo6 import diagRH

# Funcion del ejercicio

def svd_reducida(A, k="max", tol=1e-15):
    n = A.shape[0] # Cantidad de filas de A
    m = A.shape[1] # Cantidad de columnas de A

    if n >= m: # Filas >= Columnas
        B = multiplicacionMatricialConNumpy(traspuestaConNumpy(A),A)  # Llamo la matriz B, la multiplicación de A y A traspuesta. (B es simetrica)

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
            valores_singulares.append((autovalores_aux[j])**(1/2))

        hatSig = np.diag(valores_singulares)  # Armo la matriz Sigma con los valores singulares en su diagonal.
        
        hatU = np.zeros((n, len(valores_singulares)))  # Creamos la matriz hatU con una matriz de ceros de dimension n (cant. filas de A) x cant. valores singulares)

        for t in range(0, len(valores_singulares), 1):  # Calcula hatU
            Av = multiplicacionMatricialConNumpy(A,hatV[:, t])  # A * v_k
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
        B = multiplicacionMatricialConNumpy(A,A.T)  # Llamo la matriz B, la multiplicación de A y A traspuesta. (B es simetrica)

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
            valores_singulares.append((autovalores_aux[j])**(1/2))

        hatSig = np.diag(valores_singulares)  # Armo la matriz Sigma con los valores singulares en su diagonal.
        
        hatV = np.zeros((m, len(valores_singulares)))  # Creamos la matriz hatV con una matriz de ceros de dimension m (cant. columnas de A) x cant. valores singulares)

        for t in range(0, len(valores_singulares), 1):  # Calcula hatV
            A_tras_u = multiplicacionMatricialConNumpy(traspuestaConNumpy(A),hatU[:, t])  # A traspuesta * u_k. Ahora es A traspuesta porque vk = (AT * σk) / uk, es decir la columna k de la matriz V es igual a (la matriz A tras * la columna k de la matriz hatU) / el valor singular sw posicion k
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
