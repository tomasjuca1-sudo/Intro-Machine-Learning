# 07. Outliers: Z-Score y Mahalanobis

**Parte:** Outliers  
**Clases:** Clase 5  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### Cuándo sirven

Los métodos estadísticos funcionan bien para **pocas dimensiones** o datos unidimensionales.
La idea general es medir distancias relativas a la dispersión de la muestra, con lo que se
encuentran datos relativamente altos o bajos.

### Z-Score (caso unidimensional)

Cuando los datos son de una sola variable, o se hace el análisis variable por variable:

```
             x_i - xbar
Z(x_i) = -----------------
          sd(x_1, ..., x_n)
```

Regla práctica: |Z| > 3 marca un candidato a outlier bajo normalidad aproximada.

**Limitación importante.** El Z-Score usa la media y la desviación muestrales, que a su vez están
contaminadas por el outlier. Un punto muy extremo infla `sd` y se esconde a si mismo. Esto se llama
efecto de enmascaramiento (masking).

**Limitación mayor.** Aplicado variable por variable, el Z-Score no detecta outliers
**multivariados**: un punto puede tener valores normales en cada variable por separado y aun así
ser imposible dada la correlación entre ellas.

### Distancia de Mahalanobis (caso multivariado)

Corrige el problema anterior porque incorpora la matriz de covarianza:

```
d_M(x)^2 = (x - mu)^T * S^{-1} * (x - mu)
```

Interpretación: es la distancia euclidiana después de blanquear los datos, es decir, después de
quitarles la correlación y estandarizar cada dirección. Los contornos de igual distancia son las
elipses de la distribución, no círculos.

Bajo normalidad multivariada, d_M^2 se distribuye aproximadamente **chi cuadrado con p grados de
libertad**. Eso da un umbral formal: un punto es atípico si d_M^2 supera el cuantil 0.975 de una
chi cuadrado con p grados de libertad.

### Por qué se pasa a métodos locales

Ambos métodos asumen **una sola nube de datos** con un centro. Si los datos tienen varios grupos
o densidades distintas, fallan. Ese es el motivo para pasar a LOF e Isolation Forest (tema 08).

---

## Cómo se relaciona el código con el tema

El notebook hace tres cosas en orden:

1. **Simula datos normales limpios** y gráfica el histograma contra la densidad teórica.
2. **Contamina la muestra a mano**: `samples[n-1] = 12.3`. Ese es el outlier inyectado.
3. **Calcula el Z-Score** con `stats.zscore(samples)` de scipy y lo gráfica. El pico del gráfico
   es el punto contaminado.

Para Mahalanobis el notebook hace **dos versiones equivalentes**:

- Versión con scipy: `distance.mahalanobis(x, mean2, inv_cov)` dentro de una lista por comprensión,
  punto por punto.
- Versión matricial: `left = (x - mean2).T @ inv_cov` y luego `mahal = left @ (x - mean2)`. Como
  esto produce una matriz n por n, la respuesta está en `mahal.diagonal()`, porque solo interesa
  el producto de cada punto consigo mismo.

Cuidado con la orientación de la matriz en ese bloque: el código genera `x` con `.T` al final, así
que `x` queda de 2 por n. Al calcular medias y covarianza hay que ser consistente con el eje. En el
código de abajo se incluye una versión limpia con las dimensiones ordenadas.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | Z-Score sobre una muestra contaminada | Del curso, class5_book1_outlier.ipynb |
| Python | Distancia de Mahalanobis | Del curso, class5_book1_outlier.ipynb |
| Python | Mahalanobis con umbral chi cuadrado (versión ordenada) | Equivalente agregado, no viene del material del curso |
| R | Z-Score y Mahalanobis en R | Equivalente agregado, no viene del material del curso |
