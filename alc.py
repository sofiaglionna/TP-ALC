import numpy as np
import pandas as pd



### 1)

def cargarDataset(carpeta):
    traincats = np.load("template-alumnos/cats_and_dogs/train/cats/efficientnet_b3_embeddings.npy")
    traindogs = np.load("template-alumnos/cats_and_dogs/train/dogs/efficientnet_b3_embeddings.npy")

    valcats = np.load("template-alumnos/cats_and_dogs/val/cats/efficientnet_b3_embeddings.npy")
    valdogs = np.load("template-alumnos/cats_and_dogs/val/dogs/efficientnet_b3_embeddings.npy")
    
    YtGatosFila1 = np.ones((1,1000))
    YtGatosFila2 = np.zeros((1,1000))
    YtGatos = np.concatenate((YtGatosFila1, YtGatosFila2), axis=0)
    
    YtPerrosFila1 = np.zeros((1,1000))
    YtPerrosFila2 = np.ones((1,1000))
    YtPerros = np.concatenate((YtPerrosFila1, YtPerrosFila2), axis=0)

    print(traincats.shape)
    print(traincats.dtype)
    
    print(traincats)

cargarDataset()



### Ejercicio 1 ###


############# DATOS IMPORTANTES PARA ENTENDER EL EJERCICIO #############

## X_t = X_training = X_dogs + X_cats = embedding entrenados de dogs + embedding entrenados de cats. ##

### Cada embedding es un vector de dimension 1535 x 1. Cada embedding representa una imagen, donde en las 1536 filas, tiene un valor numerico ###
#### donde cada número representa una propiedad abstracta que la red aprendió a detectar como por ejemplo bordes, textura, forma, color, etc. ####

##### Cada número lo extrajo previamente EfficientNet cuando procesó la imagen, y nosotros no sabemos que representa cada valor, pero juntos forman una "firma matemática" de la imagen.



def cargarDataset(carpeta):
    # El input "carpeta" sería la carpeta "cats_and_dogs" (eso entendí yo)
    
    # Primero calculamos X_training de los embedding entrenados de perros y gatos:
    X_t_cats = np.load("carpeta/train/dogs/efficientnet_b3_embeddings.npy")  # Matriz Numpy de tamaño 1536 x N (cantidad de embedding, osea imagenes representados por un vector). Un embedding por imagen.
    X_t_dogs = np.load("carpeta/train/cats/efficientnet_b3_embeddings.npy")  # Matriz Numpy de tamaño 1536 x N (cantidad de embedding, osea imagenes representados por un vector). Un embedding por imagen.
    
    X_t = np.hstack([X_t_cats, X_t_dogs])  # "np.hstack()" apila matrices de forma horizontal, es decir que junta las columnas de cada matriz en una sola.
    
    
    
    #Ahora lo mismo, pero para los X_validation:
    X_v_cats = np.load("carpeta/val/dogs/efficientnet_b3_embeddings.npy")  # Matriz Numpy de tamaño 1536 x N (cantidad de embedding, osea imagenes representados por un vector). Un embedding por imagen.
    X_v_dogs = np.load("carpeta/val/cats/efficientnet_b3_embeddings.npy")  # Matriz Numpy de tamaño 1536 x N (cantidad de embedding, osea imagenes representados por un vector). Un embedding por imagen.
    
    X_v = np.hstack([X_v_cats, X_v_dogs])  # "np.hstack()" apila matrices de forma horizontal, es decir que junta las columnas de cada matriz en una sola.
    
    
    
    # Para Y_t e Y_v, vamos a armar una matriz de 2 (cantidad de clases posibles) x N (cantidad de embedding).
    
    # El valor de cada coordenada de fila x columna, puede ser 1 (si pertenece a esa clase) o 0 (si NO pertenece a esa clase)
    
    # La clase serían: fila 0 --> Es gato?  ;  fila 1 --> Es perro?
    
    # Ejemplo: si el embedding_1 es gato, en la fila 0 (Es gato?) va a tener valor 1, y en la fila 1 (Es perro?) va a tener valor 0. Si el embedding_2 es perro, va a tener valor 0 en fila 0, y valor 1 en fila 1.
    
    # Para armar Y_t e Y_v tenemos que saber la cantidad de imagenes que corresponden a perros y gatos asi de esa manera asignamos el valor 1 o 0 dependiendo lo que sea.
    
    # Embedding de training:
    cant_embedding_cats_t = X_t_cats.shape[1]
    cant_embedding_dogs_t = X_t_dogs.shape[1]
    
    #Embedding de validation:
    cant_embedding_cats_v = X_v_cats.shape[1]
    cant_embedding_dogs_v = X_v_dogs.shape[1]
    
    # Armo Y_t e Y_v:
    Y_t = np.zeros((2, cant_embedding_cats_t + cant_embedding_dogs_t))
    Y_v = np.zeros((2, cant_embedding_cats_v + cant_embedding_dogs_v))
    
    cant_emb_totales_t = X_t.shape[1]
    cant_emb_totales_v = X_v.shape[1]
    
    for i in range(0, cant_embedding_cats_t, 1):
        Y_t[0,i] = 1
        Y_t[1,i] = 0
    
    for j in range(cant_embedding_cats_t, cant_emb_totales_t, 1):
        Y_t[0,j] = 0
        Y_t[1,j] = 1
    
    for n in range(0, cant_embedding_cats_v, 1):
        Y_v[0,n] = 1
        Y_v[1,n] = 0
    
    for k in range(cant_embedding_cats_v, cant_emb_totales_v, 1):
        Y_v[0,k] = 0
        Y_v[1,k] = 1
    
    
    return X_t, Y_t, X_v, Y_v