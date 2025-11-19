import numpy as np
from alcModulos import traspuestaConNumpy as traspuesta, inversa, multiplicacionMatricialConNumpy as multiplicacionMatricial, calculaQR, matricesIguales, esSimetricaConTol, calculaCholesky, sustitucionHaciaAdelante, sustitucionHaciaAtras, svd_reducida

# ======================================== 
#%% 1. LECTURA DE DATOS
# ========================================

# Cada embedding es un vector de dimension 1535 x 1. Cada embedding representa una imagen que procesó EfficientNet.

### DEVUELVE: Xt, Xv, Yt, Yv
### Donde:
### Xt = 1536 filas x 2000 columnas (1000 img de gatos y 1000 img de perros)
### Xv = 1536 filas x 1000 columnas (500 img de gatos y 500 img de perros)
### Yt = 2 filas x 2000 columnas (1000 gatos y 1000 perros)
### Yv = 2 filas x 1000 columnas (500 gatos y 500 perros)

def cargarDataset(carpeta):
    # X_train:
    traincats = np.load(carpeta + "train/cats/efficientnet_b3_embeddings.npy")
    traindogs = np.load(carpeta + "train/dogs/efficientnet_b3_embeddings.npy")

    # Juntamos X_train de gatos y perros:
    Xt = np.concatenate((traincats, traindogs), axis=1)

    # X_validation:
    valcats = np.load(carpeta + "val/cats/efficientnet_b3_embeddings.npy")
    valdogs = np.load(carpeta + "val/dogs/efficientnet_b3_embeddings.npy")

    # Juntamos X_validation de gatos y perros:
    Xv = np.concatenate((valcats, valdogs), axis=1)

    #################################################
    # Creamos Y_train:
    YtCatsFila1 = np.ones((1,1000))
    YtCatsFila2 = np.zeros((1,1000))
    YtCats = np.concatenate((YtCatsFila1, YtCatsFila2), axis=0)

    YtDogsFila1 = np.zeros((1,1000))
    YtDogsFila2 = np.ones((1,1000))
    YtDogs = np.concatenate((YtDogsFila1, YtDogsFila2), axis=0)

    # Juntamos Y_trainings:
    Yt = np.concatenate((YtCats, YtDogs), axis=1)
    #dfYt = pd.DataFrame(Yt)

    # Creamos Y_validation:
    YvCatsFila1 = np.ones((1,500))
    YvCatsFila2 = np.zeros((1,500))
    YvCats = np.concatenate((YvCatsFila1, YvCatsFila2), axis=0)


    YvDogsFila1 = np.zeros((1,500))
    YvDogsFila2 = np.ones((1,500))
    YvDogs = np.concatenate((YvDogsFila1, YvDogsFila2), axis=0)

    # Juntamos Y_validations:
    Yv = np.concatenate((YvCats, YvDogs), axis=1)
    #dfYv = pd.DataFrame(Yv)

    return Xt, Yt, Xv, Yv

# ========================================
#%% 2. ECUACIONES NORMALES - Algoritmo 1
# ========================================
"""
La función se denomina pinvEcuacionesNormales(L, Y). 
-La función recibe la matriz X de los embeddings de entrenamiento,
-L la matriz de Cholesky, e Y la matriz de targets de entrenamiento. 
-La función devuelve W.

X nxp (n = 1536 x p = 2000) --->  n < p
Aplica caso b) Si rango(X) = p, n < p, entonces X+=Xt 
"""

def pinVEcuacionesNormales(X, L, Y):

    Z_traspuesta = sustitucionHaciaAdelante(L, X) 

    L_traspuesta = traspuesta(L)
    V_traspuesta = sustitucionHaciaAtras(L_traspuesta, Z_traspuesta)

    V = traspuesta(V_traspuesta)
    W_cholesky = multiplicacionMatricial(Y,V)

    return W_cholesky

# ========================================
#%% 3. DESCOMPOSICIÓN EN VALORES SINGULARES (SVD) - Algoritmo 2
# ========================================
"""
La función se denomina pinvSVD(U, S, V, Y). 
-La función recibe la matriz X de los embeddings de entrenamiento,
-las matrices U, S, V de la descomposición SVD, e Y la matriz de targets de entrenamiento. 
-La función devuelve W.

pseudoinversa de X con SVD:
- X = U·Σ·Vᵀ
- X⁺ = V·Σ⁺·Uᵀ 
- W = Y·X⁺ = Y·V·Σ⁺·Uᵀ
"""

def pinvSVD(U, S, Vt, Y):
    
    Ut = traspuesta(U)

    #Pseudoinversa de sigma:
    #invertimos dimensiones de sigma
    n = U.shape[1] #1536
    p = Vt.shape[0] #2000

    Sigma_pseudo = np.zeros((p,n))
    tol = 1e-10
    #asignamos los valores pseudoinversa de S (mismos que S pero invertidos)
    for i in range(min(p,n)):
        if abs(S[i]) > tol:
            Sigma_pseudo[i][i] = 1.0 / S[i]
        else:
            Sigma_pseudo[i][i] = 0.0

    #Hacemos V cuadrada (2000x2000) ya que para reducir tiempos, nuestra función de SVD devuelve una V de 2000x1536
    
    V_expandida = np.zeros((2000,2000))
    for i in range(2000):
        for j in range(1536):
            V_expandida[i][j] = Vt[i][j]

    VxSigma = multiplicacionMatricial(V_expandida,Sigma_pseudo) # V = pxp , Sigmapseudo = pxn , ---> VxSigma = pxn 

    pseudoX = multiplicacionMatricial(VxSigma, Ut) # VxSigma = pxn , Ut = nxn , --->  VxSigmaxU = pxn

    W_SVD = multiplicacionMatricial(Y, pseudoX) # Y = mxp , VxSigmaxU = pxn , ---> W_SVD = mxn
    
    return W_SVD

# ========================================
#%% 4. DESCOMPOSICIÓN QR - Algoritmo 3
# ========================================

#%% a) HouseHolder
"""
La función se denomina pinvHouseHolder(Q, R, Y). 
La función recibe la matriz X de los embeddings de entrenamiento, las matrices Q,R de la
descomposición QR utilizando HouseHolder, y Y la matriz de targets de entrenamiento.
La función devuelve W.
"""

def pinvHouseHolder(Q, R, Y):
    #trasponemos las 2 matrices para hacer (R^T)*(V^T) = (Q^T) y despejar V^T
    Qtraspuesta = traspuesta(Q)
    #resuelvo V * R^T = Q como R*V^T = Q^T
    Vtraspuesta= sustitucionHaciaAtras (R, Qtraspuesta)
    V = traspuesta(Vtraspuesta)
    W = multiplicacionMatricial(Y,V)
    return W

#%% b) Gram-Schmidt 
"""
La función se denomina pinvGramSchmidt(Q, R, Y). 
La función recibe la matriz X de los embeddings de entrenamiento, las matrices Q,R de la
descomposición QR utilizando GramSmidth, y Y la matriz de targets de entrenamiento. 
La función devuelve W.
"""
def pinvGramSchmidt(Q, R, Y):
    #trasponemos las 2 matrices para hacer (R^T)(V^T) = (Q^T) y despejar V^T
    Qtraspuesta = traspuesta(Q)
    #resuelvo V R^T = Q como R*V^T = Q^T
    Vtraspuesta= sustitucionHaciaAtras (R, Qtraspuesta)
    V = traspuesta(Vtraspuesta)
    W = multiplicacionMatricial(Y,V)
    return W

# ========================================
#%% 5. Pseudo-Inversa de Moore-Penrose 
# ========================================

"""
Recibe dos matrices (X, pX) y devuelve True si verifican las condiciones de Moore-Penrose (Es decir si pX es pseudo inversa de X). 
En caso contrario devuelve False.
"""

def esPseudoInverda(X, pX, tol=1e-8):
    #primera condicion: A*(A^+)*A = A
    a = multiplicacionMatricial(multiplicacionMatricial(X, pX),X) 
    #segunda condicion: (A^+)*A*(A^+) = (A^+)
    b = multiplicacionMatricial(multiplicacionMatricial(pX, X), pX) 
    #tercera condicion: (A*(A^+))^T = (A*(A^+)) (simetría)
    c = traspuesta(multiplicacionMatricial(X,pX))
    #cuarta condicion: ((A^+)*A)^T = ((A^+)A*) (simetría)
    d = traspuesta(multiplicacionMatricial(pX,X))

    if (not matricesIguales(X,a,tol = tol)) or (not matricesIguales(pX,b,tol = tol)) or not(esSimetricaConTol(c,tol = tol)) or not(esSimetricaConTol(d,tol = tol)):
        return False
    return True