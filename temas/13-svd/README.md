# 13. SVD (descomposición en valores singulares)

**Parte:** Reducción de dimensiones  
**Clases:** Clase 9  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### La descomposición

Cualquier matriz X de n por p se puede escribir como:

```
X = U Lambda V^t
```

- `U`: matriz n por n de vectores singulares izquierdos (ortonormales).
- `Lambda`: matriz diagonal con los **valores singulares**, ordenados de mayor a menor.
- `V`: matriz p por p de vectores singulares derechos (ortonormales).

### Low Rank Approximation

La fórmula central de la clase:

```
X = U Lambda V^t  aprox  U_r Lambda_r V_r^t
```

Quedarse con los primeros r valores singulares da la **mejor aproximación de rango r** de la matriz,
en el sentido de la norma de Frobenius (teorema de Eckart-Young). Es la versión matricial de lo que
PCA hace con los datos.

Relación con PCA: si X está centrada, los vectores singulares derechos V son exactamente los
eigenvectores de la matriz de covarianza, y los valores singulares al cuadrado son proporcionales a
los eigenvalores. Por eso `prcomp` en R usa SVD internamente.

### Aplicación: information retrieval

Es el ejemplo del curso. Se construye la **matriz término documento**: filas son términos, columnas
son documentos, y la entrada (i, j) es la frecuencia del término i en el documento j.

Problemas de esa matriz: es enorme, es muy dispersa (casi todo ceros) y sufre de sinonimia
(palabras distintas con el mismo significado) y polisemia (una palabra con varios significados).

Al aplicar SVD y quedarse con rango r se obtiene **Latent Semantic Indexing**: los documentos y las
consultas se representan en un espacio de conceptos latentes de dimensión r, donde sinónimos quedan
cerca.

### Búsqueda por similitud de coseno

Una consulta q se proyecta al espacio reducido y se compara con cada documento usando el coseno del
ángulo:

```
                 < v_j , q_proyectado >
cos(v_j, q) = ------------------------------
                ||v_j|| * ||q_proyectado||
```

El coseno va de -1 a 1 (de 0 a 1 con frecuencias no negativas). Se usa el coseno en vez de la
distancia euclidiana porque **no depende de la longitud del documento**, solo de la dirección, es
decir, de la composición temática.

---

## Cómo se relaciona el código con el tema

Código del curso, en R, con el paquete `tm` para minería de texto.

**Paso 1, construir la matriz.**

```
tdm <- TermDocumentMatrix(crude, control = list(removePunctuation = TRUE, stopwords = TRUE))
tt = as.matrix(tdm)
```

`crude` son 20 artículos sobre petróleo. `removePunctuation` y `stopwords` son limpieza estándar de
texto. El resultado `tt` es la matriz término documento.

**Paso 2, descomponer.**

```
ss = svd(tt)
ss$d   # valores singulares
ss$u   # vectores izquierdos
ss$v   # vectores derechos
```

La línea `tt - ss$u %*% diag(ss$d) %*% t(ss$v)` es una **verificación**: debe dar prácticamente cero.
Comprueba que la descomposición reconstruye la matriz original.

**Paso 3, aproximación de rango bajo.** El código lo hace de dos formas equivalentes y luego las
resta para mostrar que coinciden:

- Truncando las matrices: `ttk = ss$u[,1:10] %*% diag(ss$d[1:10]) %*% t(ss$v)[1:10,]`
- Poniendo ceros en los valores singulares sobrantes:
  `ttk3 = ss$u %*% diag(c(ss$d[1:10], rep(0,10))) %*% t(ss$v)`

`ttk - ttk3` da cero. Esa equivalencia es la misma idea del truco de los ceros en la reconstrucción
de PCA (tema 09).

**Paso 4, consulta.** Se simula una consulta binaria `q` (que términos aparecen), se proyecta al
espacio reducido con `qq = diag(dd) %*% t(uu) %*% q` y se calculan los cosenos contra cada documento
con la fórmula vectorizada del final.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | SVD y aproximación de rango bajo con numpy | Equivalente agregado. El código del curso está en R |
| Python | Matriz término documento, LSI y búsqueda por coseno | Equivalente agregado. Replica el ejemplo de information retrieval del curso |
| R | SVD e information retrieval (código del curso) | Del curso, code3_cass9_SVD.txt |
| R | SVD básico y relación con PCA | Equivalente agregado, no viene del material del curso |
