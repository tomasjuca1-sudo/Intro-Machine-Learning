# 12. Kernel PCA

**Parte:** Reducción de dimensiones  
**Clases:** Clase 9  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### El problema que resuelve

PCA es **lineal**: solo puede encontrar direcciones que sean combinaciones lineales de las variables
originales. Si la estructura de los datos es curva (dos esferas concentricas, una espiral, dos
anillos), PCA no la separa por más componentes que se usen.

### La idea

Se lleva cada observación a un espacio de mayor dimensión mediante una función phi, y se hace PCA
**allí**:

```
x  ->  phi(x)   ->  PCA en el espacio transformado
```

En el espacio transformado la estructura puede volverse lineal y por lo tanto separable.

### El truco del kernel

Calcular phi(x) explícitamente sería carísimo o imposible (el espacio puede ser infinito
dimensional). El truco: PCA solo necesita **productos punto**, y un kernel los calcula sin construir
phi:

```
K(x, y) = < phi(x) , phi(y) >
```

Se construye la matriz de kernel K de n por n, se centra, y se hace descomposición espectral sobre
ella. Las componentes salen de los eigenvectores de K.

### Kernels comunes

| Kernel | Fórmula | Parámetro |
|---|---|---|
| Gaussiano o RBF (`rbfdot`) | exp(-sigma * ||x - y||^2) | sigma |
| Polinomial (`polydot`) | (escala * <x,y> + offset)^grado | grado |
| Lineal (`vanilladot`) | <x, y> | ninguno, equivale a PCA normal |
| Tangente hiperbólica (`tanhdot`) | tanh(escala * <x,y> + offset) | escala |

El **sigma** del kernel gaussiano es el parámetro crítico. Sigma grande hace el kernel muy local
(cada punto se vuelve su propio grupo), sigma pequeño se acerca al caso lineal. En el código del
curso se usan valores distintos según los datos: 0.2 para iris, 1.1 para los datos circulares y 0.1
para measure.

### Limitación importante

KPCA **no tiene reconstrucción directa** al espacio original. Se puede proyectar hacia adelante pero
volver de Z a X es un problema aparte (pre-image problem). PCA clásico sí reconstruye. Esto importa
si el objetivo es comprimir y luego descomprimir.

---

## Cómo se relaciona el código con el tema

La función del curso es `kpca` del paquete **kernlab** en R.

```
kpc <- kpca(~., data = iris[-test, -5], kernel = "rbfdot",
            kpar = list(sigma = 0.2), features = 2, eta = 0.001, maxiter = 65)
```

Argumentos:

- `~.` es la fórmula: usar todas las columnas del data frame.
- `kernel = "rbfdot"` escoge el kernel gaussiano.
- `kpar = list(sigma = 0.2)` fija su parámetro.
- `features = 2` es el número de componentes (el d de la teoría).
- `eta` y `maxiter` controlan la convergencia numérica.

Salidas:

- `pcv(kpc)`: los vectores de componentes principales en el espacio del kernel.
- `predict(kpc, datos_nuevos)`: **proyecta datos nuevos** sobre las componentes. Esto es lo que
  permite el ejercicio del notebook de ajustar con una parte de iris y luego embeber los puntos
  restantes (`emb <- predict(kpc, iris[test, -5])`).

El ejemplo más didáctico del código es el de los **datos circulares**: se generan dos casquetes
esféricos concentricos en 3D (`z = sqrt(7 - x^2 - y^2)` y `z = sqrt(3 - x^2 - y^2)`), se proyectan
primero con `princomp` (PCA lineal) y luego con `kpca`. Con PCA los dos grupos quedan mezclados,
con KPCA quedan separados. Ese contraste es el punto de la clase.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | KernelPCA con sklearn | Equivalente agregado. El código del curso está en R |
| Python | Comparar kernels y calibrar gamma | Equivalente agregado, no viene del material del curso |
| R | kpca con kernlab (código del curso) | Del curso, code2_class9_KPCA.txt |
