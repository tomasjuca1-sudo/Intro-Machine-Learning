# 09. PCA (componentes principales)

**Parte:** Reducción de dimensiones  
**Clases:** Clase 6, Clase 7, Clase 8  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### El problema general de reducción de dimensiones

```
Input :  matriz X de n x p.  Se observan p variables X1, ..., Xp
Output:  matriz Z de n x d con d < p.  Se representan los datos en d variables Z1, ..., Zd
```

El criterio de "mejor representación posible" del curso es **conservar las relaciones entre los
puntos**, es decir, las distancias:

```
Dist(x_i, x_j)  aprox  Dist(z_i, z_j)   para todo i, j
```

Por qué se usa: entender mejor los datos en grandes dimensiones (feature extraction, variables
latentes, graficar en 2D o 3D), manipular datos de forma más eficiente (compresión de imágenes,
cómputos más rápidos) y eliminar información redundante (modelos supervisados con menos variables).

### PCA como método lineal

En los métodos lineales la transformación es una multiplicación por una matriz:

```
Z = X A       con A en R^{p x d}
```

PCA escoge A de forma que las nuevas variables (componentes) tengan **máxima varianza** y sean
**ortogonales** entre si.

### Solución por descomposición espectral

Sea S la matriz de covarianza muestral de X. Se resuelve:

```
S w = lambda w
```

- El **primer componente principal** es el eigenvector w_1 asociado al eigenvalor más grande.
- La **varianza del componente j** es exactamente el eigenvalor lambda_j.
- La **proporción de varianza explicada** por el componente j es lambda_j / sum(lambda).

Los **scores** son las coordenadas de los datos en la nueva base, calculadas sobre datos centrados:

```
Z = (X - media) W
```

### Reconstrucción

Con d componentes se puede reconstruir una aproximación de los datos originales:

```
X_aprox = Z_d W_d^T + media
```

Cuantos más componentes, mejor la reconstrucción. Con d = p la reconstrucción es exacta. Este es
el principio de la compresión de imágenes del ejemplo de dígitos.

### Decisiones prácticas

- **Centrar siempre.** PCA sobre datos no centrados mide algo distinto.
- **Escalar (usar la matriz de correlación) si las unidades difieren.** Con `princomp` en R se usa
  `cor = TRUE`, en Python se estandariza antes. Sin escalar, la variable con mayor varianza numérica
  domina el primer componente.
- **Cuántos componentes**: gráfico de codo de los eigenvalores (`plot(pca$sdev^2, type="b")`),
  o varianza acumulada superior a un porcentaje, o la regla de Kaiser (eigenvalores mayores a 1
  sobre la matriz de correlación).
- **Biplot**: muestra al tiempo los scores de las observaciones y las cargas (loadings) de las
  variables originales. Sirve para interpretar qué significa cada componente.

---

## Cómo se relaciona el código con el tema

El código del curso para PCA está en **R**. Hace el procedimiento dos veces, a mano y con la función.

**Versión manual, paso a paso:**

| Concepto | Línea |
|---|---|
| Matriz de covarianza S | `S = var(X)` |
| Descomposición espectral | `eigen(S)` |
| Primer eigenvector w_1 | `w = eigen(S)$vectors[,1]` |
| Primer eigenvalor | `lambda = eigen(S)$values[1]` |
| Datos centrados | `XX = as.matrix(X - medias)` |
| Primer componente (scores) | `Z = XX %*% w` |
| Reconstrucción | `ZZ = cbind(Z*w[1], Z*w[2])` y se le suman las medias |

**Versión con la función:** `princomp(X, scores = TRUE)`.

- `summary(pca, loadings = TRUE)` da la varianza explicada y las cargas.
- `pca$sdev^2` son los eigenvalores. Graficarlos es el **gráfico de codo**.
- `pca$scores` son las nuevas coordenadas Z.
- `sqrt(lambda)` coincide con la primera entrada de `pca$sdev`, y
  `lambda/sum(eigen(S)$values)` es la proporción de varianza explicada. El código hace esa
  verificación a propósito.

En el ejemplo de dígitos, la reconstrucción con k componentes se arma pegando ceros:
`ZZ = Z %*% rbind(t(w)[1:k,], matriz de ceros)`. Eso equivale a anular los componentes que no se
usan. Luego se vuelve a sumar la media y se gráfica la imagen reconstruida contra la original.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | PCA con sklearn (equivalente del código en R) | Equivalente agregado. El código de PCA del curso está en R |
| Python | PCA a mano con descomposición espectral y reconstrucción | Equivalente agregado. Replica el procedimiento manual del código en R |
| Python | Biplot en Python | Equivalente agregado, no viene del material del curso |
| R | PCA paso a paso y con princomp (código del curso) | Del curso, code1_class7_ejemplo1-pca.txt |
| R | PCA sobre dígitos manuscritos y reconstrucción de imágenes | Del curso, code2_class7_ejemplo2-pca.txt |
| R | prcomp: la alternativa a princomp | Equivalente agregado, no viene del material del curso |
