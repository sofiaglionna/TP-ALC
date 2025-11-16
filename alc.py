import numpy as np
import pandas as pd
from Modulos import traspuestaConNumpy as traspuesta, inversa, multiplicacionMatricialConNumpy as multiplicacionMatricial, calculaQR, matricesIguales, esSimetricaConTol, calculaCholesky, sustitucionHaciaAdelante, sustitucionHaciaAtras, svd_reducida

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

# X_train:
traincats = np.load("template-alumnos/cats_and_dogs/train/cats/efficientnet_b3_embeddings.npy")
traindogs = np.load("template-alumnos/cats_and_dogs/train/dogs/efficientnet_b3_embeddings.npy")
dftraincats = pd.DataFrame(traincats) #Spyder
dftraindogs = pd.DataFrame(traindogs) #Spyder
# Juntamos X_train de gatos y perros:
Xt = np.concatenate((traincats, traindogs), axis=1)
dfXt = pd.DataFrame(Xt) #Spyder

# X_validation:
valcats = np.load("template-alumnos/cats_and_dogs/val/cats/efficientnet_b3_embeddings.npy")
valdogs = np.load("template-alumnos/cats_and_dogs/val/dogs/efficientnet_b3_embeddings.npy")
dfvalcats = pd.DataFrame(valcats) #Spyder
dfvaldogs = pd.DataFrame(valdogs) #Spyder
# Juntamos X_validation de gatos y perros:
Xv = np.concatenate((valcats, valdogs), axis=1)
dfXv = pd.DataFrame(Xv) #Spyder

#################################################
# Creamos Y_train:
YtCatsFila1 = np.ones((1,1000))
YtCatsFila2 = np.zeros((1,1000))
YtCats = np.concatenate((YtCatsFila1, YtCatsFila2), axis=0)
dfYtCats = pd.DataFrame(YtCats) #Spyder

YtDogsFila1 = np.zeros((1,1000))
YtDogsFila2 = np.ones((1,1000))
YtDogs = np.concatenate((YtDogsFila1, YtDogsFila2), axis=0)
dfYtDogs = pd.DataFrame(YtDogs) #Spyder
# Juntamos Y_trainings:
Yt = np.concatenate((YtCats, YtDogs), axis=1)
dfYt = pd.DataFrame(Yt) #Spyder

# Creamos Y_validation:
YvCatsFila1 = np.ones((1,500))
YvCatsFila2 = np.zeros((1,500))
YvCats = np.concatenate((YvCatsFila1, YvCatsFila2), axis=0)
dfYvCats = pd.DataFrame(YvCats) #Spyder

YvDogsFila1 = np.zeros((1,500))
YvDogsFila2 = np.ones((1,500))
YvDogs = np.concatenate((YvDogsFila1, YvDogsFila2), axis=0)
dfYvDogs = pd.DataFrame(YvDogs) #Spyder

# Juntamos Y_validations:
Yv = np.concatenate((YvCats, YvDogs), axis=1)
dfYv = pd.DataFrame(Yv) #Spyder

# ========================================
#%% 2. ECUACIONES NORMALES - Algoritmo 1
# ========================================
"""
La función se denomina pinvEcuacionesNormales(L, Y). 
-La función recibe la matriz X de los embeddings de entrenamiento, ---> Esto no lo entiendo pq no se le pasa como parámetro "X"
-L la matriz de Cholesky, y Y la matriz de targets de entrenamiento. 
-La función devuelve W.

X pertenece a (n = 1536 x p = 2000) --->  n < p
Aplica caso b) Si rango(X) = p, n < p, entonces X+=Xt 
"""
A = multiplicacionMatricial(Xt, traspuesta(Xt))

L = calculaCholesky(A) #tolerancia 0 funciona?
# A = LL^T
traspuestaX = traspuesta(Xt)

def pinVEcuacionesNormales(X, L, Y):
    # L^T Z^T = X
    L_traspuesta = traspuesta(L)
    Z_traspuesta = sustitucionHaciaAtras(L_traspuesta, X) 
    # L V^T = Z^T
    V_traspuesta = sustitucionHaciaAdelante(L, Z_traspuesta)
    
    # Finalmente resolvemos W = Y × V:
    # Donde V = pseudo inversa de X
    V = traspuesta(V_traspuesta)
    W_cholesky = multiplicacionMatricial(Y, V)
    
    return W_cholesky


# ========================================
#%% 3. DESCOMPOSICIÓN EN VALORES SINGULARES (SVD) - Algoritmo 2
# ========================================
"""
La función se denomina pinvSVD(U, S, V, Y). 
-La función recibe la matriz X de los embeddings de entrenamiento,
-las matrices U, S, V de la descomposición SVD, e Y la matriz de targets de entrenamiento. 
-La función devuelve W.

Parámetros:
-----------
U : np.array
    Matriz U de la descomposición SVD (nxr) donde r = rango(X)
S : np.array
    Matriz diagonal Σ con valores singulares (rxr)
V : np.array
    Matriz V de la descomposición SVD (pxr)
Y : np.array
    Matriz de targets (mxp)

- X = U·Σ·Vᵀ es la descomposición SVD
- X⁺ = V·Σ⁺·Uᵀ es la pseudoinversa
- W = Y·X⁺ = Y·V·Σ⁺·Uᵀ
"""
# X pertenece a (n = 1536 x p = 2000) --->  n < p

# Desc de X
U, S, Vt = svd_reducida(Xt, k="max")

def pinvSVD(U, S, Vt, Y):
    
    V = traspuesta(Vt) # svd devuelve V traspuesta asi que la trasponemos de nuevo para obtener V.

    Ut = traspuesta(U)

    #Pseudoinversa de sigma:
    diag_S = np.diag(S)

    n = Ut.shape[0] #1536
    r = S.shape[0] 

    diag_S_inv = np.zeros(r)
    tol = 1e-10
    
    for i in range(r):
        if abs(diag_S[i]) > tol:
            diag_S_inv[i] = 1.0 / diag_S[i]
        else:
            diag_S_inv[i] = 0.0
    
    # Creamos la pseudoinversa de sigma como matriz diagonal
    Sigma_pseudo = np.diag(diag_S_inv) #pxn

    VxSigma = multiplicacionMatricial(V,Sigma_pseudo) # V = pxp , Sigmapseudo = pxn , ---> VxSigma = pxn 

    VxSigmaxU = multiplicacionMatricial(VxSigma, Ut) # VxSigma = pxn , Ut = nxn , --->  VxSigmaxU = pxn

    W_SVD = multiplicacionMatricial(Y, VxSigmaxU) # Y = mxp , VxSigmaxU = pxn , ---> W_SVD = mxn

    return W_SVD

# ========================================
#%% 4. DESCOMPOSICIÓN QR - Algoritmo 3
# ========================================

#%% a) HouseHolder
"""
La función se denomina pinvHouseHolder(Q, R, Y). La función recibe
la matriz X de los embeddings de entrenamiento, las matrices Q,R de la
descomposición QR utilizando HouseHolder, y Y la matriz de targets de entrenamiento.
La función devuelve W.
"""

#Calculamos Q y R de X traspuesta que son las que necesitamos para el algoritmo 3
XtTraspuesta = traspuesta(Xt)
QyRHH = calculaQR(XtTraspuesta, "RH")
QHH = QyRHH[0]
RHH = QyRHH[1]
#Q.shape = 2000x2000
#R.shape = 2000x1536

#me quedo con las filas y columnas utiles de QR
filas, columnas = XtTraspuesta.shape
QHH = QHH[0:filas,0:columnas]
RHH = RHH[0:columnas,0:columnas]

def pinvHouseHolder(Q, R, Y):
    #trasponemos las 2 matrices para hacer (R^T)*(V^T) = (Q^T) y despejar V^T
    Rtraspuesta = traspuesta(R)
    Qtraspuesta = traspuesta(Q)
    #resolvemos el sistema (R ya es triangular inferior)
    Vtraspuesta= sustitucionHaciaAdelante (Rtraspuesta, Qtraspuesta)
    V = traspuesta(Vtraspuesta)
    W = multiplicacionMatricial(Y,V)
    return W

#%% b) Gram-Schmidt 
"""
La función se denomina pinvGramSchmidt(Q, R, Y). La función
recibe la matriz X de los embeddings de entrenamiento, las matrices Q,R de la
descomposición QR utilizando GramSmidth, y Y la matriz de targets de
entrenamiento. La función devuelve W.
"""

QyRGS = calculaQR(XtTraspuesta, "GS")
QGS = QyRGS[0]
RGS = QyRGS[1]
#Q.shape = 2000x2000
#R.shape = 2000x1536

#me quedo con las filas y columnas utiles de QR

QGS = QGS[0:filas,0:columnas]
RGS = RGS[0:columnas,0:columnas]
###########################################
#el enunciado dice gram schmidt clasico, nosotros usamos el modificado. Lo cambiamos?
##########################################
def pinvGramSchmidt(Q, R, Y):
    #trasponemos las 2 matrices para hacer (R^T)*(V^T) = (Q^T) y despejar V^T
    Rtraspuesta = traspuesta(R)
    Qtraspuesta = traspuesta(Q)
    #resolvemos el sistema (R ya es triangular inferior)
    Vtraspuesta= sustitucionHaciaAdelante (Rtraspuesta, Qtraspuesta)
    V = traspuesta(Vtraspuesta)
    W = multiplicacionMatricial(Y,V)
    return W

# ========================================
#%% 5. Pseudo-Inversa de Moore-Penrose 
# ========================================

"""
Recibe dos matrices y devuelva True si verifican las condiciones de Moore-Penrose. 
En caso contrario devolver False.
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