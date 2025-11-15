import numpy as np
import pandas as pd

### Ejercicio 1 ###

############# DATOS IMPORTANTES PARA ENTENDER EL EJERCICIO #############

### Cada embedding es un vector de dimension 1535 x 1. Cada embedding representa una imagen, donde en las 1536 filas, tiene un valor numerico ###
#### donde cada número representa una propiedad abstracta que la red aprendió a detectar como por ejemplo bordes, textura, forma, color, etc. ####

##### Cada número lo extrajo previamente EfficientNet cuando procesó la imagen, y nosotros no sabemos que representa cada valor, pero juntos forman una "firma matemática" de la imagen.

#%%
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

#%%
