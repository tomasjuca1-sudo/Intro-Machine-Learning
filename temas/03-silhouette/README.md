# 03. Silhouette y métricas de ajuste

**Parte:** Clustering  
**Clases:** Clase 3, Clase 4  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### Para qué sirve

El algoritmo de clustering optimiza su propio criterio interno, así que el SSE no sirve para
comparar K distintos (siempre baja cuando K sube). Las métricas de ajuste son **externas al
algoritmo** y permiten comparar soluciones con distinto número de grupos o con distintos métodos.

### Silhouette

Se calcula **para cada observación** x_i:

```
a(i) = distancia promedio de x_i a los demás puntos de SU cluster  C_I
b(i) = distancia promedio de x_i a los puntos del cluster más cercano distinto  C_k

              b(i) - a(i)
s(i) = ---------------------
          max{ a(i) , b(i) }
```

Propiedades:

- Siempre se cumple **-1 <= s(i) <= 1**.
- s(i) cerca de 1: el punto está mucho más cerca de su grupo que del vecino. Buena asignación.
- s(i) cerca de 0: el punto está en la frontera entre dos grupos.
- s(i) negativo: el punto está **más cerca de otro grupo** que del propio. Mala asignación.

El Silhouette global es el promedio de los s(i). Para escoger K se corre el algoritmo con varios
valores y se toma el K con mayor Silhouette promedio.

El gráfico de Silhouette ordena los s(i) dentro de cada cluster. Un cluster con barras cortas o
negativas es un cluster mal formado.

### Otras métricas

**Calinski-Harabasz.** Razón entre dispersión entre grupos y dispersión dentro de grupos,
ajustada por grados de libertad. Es una idea tipo F de ANOVA.

```
CH = [ BSS / (K-1) ] / [ WSS / (n-K) ]
```

Se busca el **máximo**.

**Davies-Bouldin.** Para cada par de clusters compara la dispersión interna contra la separación
de centros, y promedia el peor caso de cada cluster. Se busca el **mínimo**.

### Regla práctica para el parcial

| Métrica | Dirección buena | Rango |
|---|---|---|
| Silhouette | máximo | -1 a 1 |
| Calinski-Harabasz | máximo | 0 a infinito |
| Davies-Bouldin | mínimo | 0 a infinito |

---

## Cómo se relaciona el código con el tema

Las tres métricas se calculan sobre **los mismos dos argumentos**: la matriz de datos y el vector
de etiquetas. No necesitan el objeto del modelo.

```
silhouette_score(Xs, y_kmeans_s)
```

Importante: se pasan los datos **en la misma escala en la que se corrió el algoritmo**. Si
agrupaste sobre `Xs` escalado, la métrica va sobre `Xs`, no sobre `X`.

Para el gráfico se usa `SilhouetteVisualizer` de yellowbrick, que sí recibe el objeto del modelo
porque necesita reajustarlo. Si yellowbrick no está disponible, `silhouette_samples` de sklearn
devuelve el vector de s(i) individuales y el gráfico se arma a mano.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | Las tres métricas en una línea cada una | Del curso, class2_book1_kmeans.ipynb |
| Python | Gráfico de Silhouette con yellowbrick | Del curso, class2_book1_kmeans.ipynb |
| Python | Escoger K comparando Silhouette (sin yellowbrick) | Equivalente agregado, no viene del material del curso |
| R | Silhouette en R con el paquete cluster | Equivalente agregado, no viene del material del curso |
