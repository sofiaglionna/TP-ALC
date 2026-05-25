# Red Neuronal Lineal con métodos de Pseudo-Inversa
## Clasificación de imágenes (Cats vs Dogs) con Transfer Learning

Trabajo Práctico de Álgebra Lineal Computacional — FCEyN, UBA — 2do Cuatrimestre 2025

### Objetivo
Implementar desde cero (sin numpy.linalg) cuatro algoritmos para resolver el problema
de mínimos cuadrados W = Y·X⁺, aplicados a la capa final de una red neuronal lineal
para clasificación de imágenes vía Transfer Learning con EfficientNet-b3.

### Algoritmos implementados
- **Ecuaciones Normales con Cholesky** — para sistemas sobre-determinados
- **SVD reducida** — descomposición en valores singulares con particionado
- **QR con Householder** — reflexiones ortogonales
- **QR con Gram-Schmidt clásico** — ortogonalización iterativa

### Resultados

| Método | Tiempo total | Accuracy |
|---|---|---|
| Cholesky | 118 min | 68.4% |
| SVD | 101 min | 68.1% |
| QR (Householder) | 55 min | 68.4% |
| QR (Gram-Schmidt) | **27 min** | 68.4% |

**Conclusión:** Gram-Schmidt resultó 4.4× más rápido que Cholesky manteniendo accuracy
equivalente. Los tiempos son elevados por la implementación propia (sin numpy) de
operaciones matriciales, exigida por la consigna.

### Nota sobre accuracy
El 68% refleja la limitación esperada de una red lineal sin función de activación
no lineal: una capa fully-connected lineal no puede aprovechar plenamente embeddings
de 1536 dimensiones. El foco del TP era comparar métodos de pseudo-inversa, no
maximizar accuracy.

### Stack
Python, NumPy (solo para tipos de arreglo), Jupyter, EfficientNet-b3 (pre-entrenado, para embeddings)

### Equipo
- Sofía Glionna
- Felix Soriano
- Pedro Soldatich
- Geronimo Gabriel Pacheco Parrondo
