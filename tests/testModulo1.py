# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 14:40:47 2025

@author: Usuario
"""
#Para usar esto en spyder hay que cambiar desde donde se ejecuta, la direccion superior derecha
import numpy as np
from Modulos.modulo1 import error, error_relativo, matricesIguales

def sonIguales(x, y, atol=1e-8):
    return np.allclose(error(x, y), 0, atol=atol)

# --- Tests unitarios con assert ---

# Pruebas de sonIguales
assert(not sonIguales(1, 1.1))
assert(sonIguales(1, 1 + np.finfo('float64').eps))
assert(not sonIguales(1, 1 + np.finfo('float32').eps))
assert(not sonIguales(np.float16(1), np.float16(1) + np.finfo('float32').eps))
assert(sonIguales(np.float16(1), np.float16(1) + np.finfo('float16').eps, atol=1e-3))

# Pruebas de error_relativo
assert(np.allclose(error_relativo(1, 1.1), 0.1))
assert(np.allclose(error_relativo(2, 1), 0.5))
assert(np.allclose(error_relativo(-1, 1), 2))
assert(np.allclose(error_relativo(1, -1), 2))

# Pruebas de matricesIguales
assert(matricesIguales(np.diag([1, 1]), np.eye(2)))
assert(matricesIguales(np.linalg.inv(np.array([[1, 2], [3, 4]])) @ np.array([[1, 2], [3, 4]]), np.eye(2)))
assert(not matricesIguales(np.array([[1, 2], [3, 4]]).T, np.array([[1, 2], [3, 4]])))

print("✅ Todos los tests pasaron correctamente.")
