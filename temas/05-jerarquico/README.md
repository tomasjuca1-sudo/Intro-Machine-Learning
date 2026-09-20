# 05. Clustering jerárquico

**Parte:** Clustering  
**Clases:** Clase 3, Clase 4  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### Qué hace distinto

Los métodos jerárquicos no buscan tanto una partición final en K grupos, sino **mostrar el proceso**
por el cual las observaciones se van agregando (aglomerativo) o separando (divisivo). El resultado
es un árbol, no una asignación única.

En el caso aglomerativo: cada punto empieza como su propio cluster, y en cada paso se unen los dos
clusters más cercanos, hasta que queda uno solo.

### Distancia entre clusters (linkage)

Es el concepto central. Se necesita una distancia entre **grupos**, no entre puntos:

| Linkage | Definición | Comportamiento |
|---|---|---|
| `single` (simple) | distancia **mínima** entre puntos de los dos clusters | encadena, detecta formas alargadas, sensible a ruido |
| `complete` (completa) | distancia **máxima** entre puntos de los dos clusters | grupos compactos y de tamaño parecido |
| `average` (promedio) | distancia **promedio** entre todos los pares | intermedio entre los dos anteriores |
| `ward` | minimiza el incremento del WSS al unir | grupos de varianza parecida, se acerca a K-Means |

### Características importantes

1. El resultado **depende de la escala**, igual que K-Means.
2. Es muy rápido de implementar.
3. Da información adicional: crea una relación de familias entre los puntos, muy útil para la
   representación visual.
4. Es **más robusto que K-Means** a los datos atípicos.

### Problemas

- **Problema 1**: el resultado depende mucho de la distancia (linkage) seleccionada.
- **Problema 2**: el método de estimación puede dar resultados no muy buenos en términos de
  coherencia o separación entre clusters, porque es un método greedy que nunca deshace una unión.

### Dendrograma

El eje vertical es la distancia a la que se unieron dos grupos. Cortar el dendrograma a una altura
h da una partición. Un salto grande de altura entre dos uniones sugiere un buen número de clusters.

---

## Cómo se relaciona el código con el tema

El código usa **dos librerías distintas para dos cosas distintas**, y conviene no confundirlas:

1. `AgglomerativeClustering` de sklearn: **produce las etiquetas** cuando ya decidiste cuántos
   grupos quieres. Parámetros: `n_clusters` y `linkage`.

2. `linkage` y `dendrogram` de scipy: **producen el árbol** y lo dibujan. `linkage(X, method=...,
   metric=...)` devuelve la matriz de fusiones, y `dendrogram()` la gráfica.

El parámetro `method` de scipy es el mismo concepto que `linkage` de sklearn. Cambiar entre
`'single'`, `'complete'`, `'average'` y `'ward'` es el ejercicio típico del parcial: con `single`
sobre iris los grupos quedan muy desbalanceados por el encadenamiento.

Nota: con `method='ward'` la métrica debe ser euclidiana.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | Aglomerativo y dendrograma | Del curso, class4_book1_hierarquical.ipynb |
| Python | Comparar los cuatro linkages y cortar el árbol | Equivalente agregado, no viene del material del curso |
| R | hclust, dendrograma y corte del árbol | Equivalente agregado, no viene del material del curso |
