# 04. K-Medoides

**Parte:** Clustering  
**Clases:** Clase 3  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### Qué cambia respecto a K-Means

Se reemplaza el promedio del grupo por el **medoide**: la observación de la muestra que es más
central dentro del grupo. El centro deja de ser un punto artificial y pasa a ser un dato real.

```
medoide_k = argmin_{x_j en cluster k}  sum_{x_i en cluster k} d(x_i, x_j)
```

### Los dos efectos que gana

1. **Robustez.** Como el centro es una observación y no un promedio, un outlier ya no arrastra el
   centro del grupo. K-Medoides es notablemente más resistente a datos atípicos que K-Means.

2. **Generalización a coordenadas no euclidianas.** El algoritmo solo necesita la **matriz de
   distancias** entre pares, no las coordenadas. Eso permite usarlo con variables categóricas,
   mixtas o con distancias no métricas. K-Means no puede hacer esto porque necesita calcular un
   promedio, y el promedio de categorías no existe.

### Distancia de Gower

Es la distancia estándar para datos mixtos (continuos, ordinales, binarios, nominales). Normaliza
cada variable a su rango y promedia las contribuciones. Es lo que habilita usar K-Medoides sobre
bases con variables categóricas.

### Costo

El algoritmo exacto (PAM) es más costoso que K-Means porque evalúa intercambios de medoides.
`fasterpam` es una implementación eficiente de PAM.

---

## Cómo se relaciona el código con el tema

El punto clave del código: **la entrada no es X, es la matriz de distancias**.

```
dista_s = distance_matrix(Xs, Xs)     # matriz n x n
c = kmedoids.fasterpam(dista_s, 3)    # recibe distancias, no datos
```

Esto es lo que hace que el método se pueda generalizar. Cambiando la forma de calcular la matriz
de distancias, se cambia el tipo de datos que el algoritmo acepta, sin tocar el algoritmo:

- Datos continuos: `distance_matrix(Xs, Xs)` (euclidiana).
- Datos mixtos o categóricos: `gower.gower_matrix(W)`.

Salidas del objeto `c`:

- `c.loss`: valor de la función objetivo, o sea la suma de distancias a los medoides.
- `c.labels`: etiqueta de grupo de cada observación.
- `c.medoids`: **índices** de las observaciones que quedaron como medoides. Son posiciones dentro
  de la muestra original, no coordenadas.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | K-Medoides con distancia euclidiana | Del curso, class3_book1_kmedoids.ipynb |
| Python | K-Medoides con distancia de Gower (datos no euclidianos) | Del curso, class3_book1_kmedoids.ipynb |
| Python | Alternativa con sklearn_extra | Equivalente agregado, no viene del material del curso |
| R | PAM y distancia de Gower en R | Equivalente agregado, no viene del material del curso. La función daisy con metric gower si aparece en el código de MDS del curso |
