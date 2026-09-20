# Introducción al Machine Learning (IIND3118) - Repositorio de parcial

Material organizado por tema para consulta rápida durante el parcial con libro abierto.
Universidad de los Andes, segundo semestre 2026. Profesor Carlos Felipe Valencia.

## Cómo usarlo durante el parcial

Abre **[`index.html`](index.html)** en el navegador. Funciona sin internet, con pestañas por tema,
sub pestañas de explicación, Python y R, buscador y botón de copiar en cada bloque.

Atajos del visor: `/` enfoca el buscador, las flechas arriba y abajo cambian de tema,
`1` `2` `3` cambian de sub pestaña y `Esc` limpia la búsqueda. El buscador ignora las tildes,
así que da igual escribir "seleccion" o "selección".

Si prefieres navegar por archivos, cada tema tiene su carpeta en `temas/` con tres archivos:
`README.md` con la explicación, `codigo.py` y `codigo.R`.

## Índice de temas

| # | Parte | Tema | Python | R | Clases |
|---|---|---|---|---|---|
| 01 | Fundamentos | [Axioma de los datos y tipos de aprendizaje](temas/01-fundamentos/README.md) | [py](temas/01-fundamentos/codigo.py) | [R](temas/01-fundamentos/codigo.R) | Clase 1, Clase 10 |
| 02 | Clustering | [K-Means](temas/02-kmeans/README.md) | [py](temas/02-kmeans/codigo.py) | [R](temas/02-kmeans/codigo.R) | Clase 2, Clase 3 |
| 03 | Clustering | [Silhouette y métricas de ajuste](temas/03-silhouette/README.md) | [py](temas/03-silhouette/codigo.py) | [R](temas/03-silhouette/codigo.R) | Clase 3, Clase 4 |
| 04 | Clustering | [K-Medoides](temas/04-kmedoids/README.md) | [py](temas/04-kmedoids/codigo.py) | [R](temas/04-kmedoids/codigo.R) | Clase 3 |
| 05 | Clustering | [Clustering jerárquico](temas/05-jerarquico/README.md) | [py](temas/05-jerarquico/codigo.py) | [R](temas/05-jerarquico/codigo.R) | Clase 3, Clase 4 |
| 06 | Clustering | [DBSCAN](temas/06-dbscan/README.md) | [py](temas/06-dbscan/codigo.py) | [R](temas/06-dbscan/codigo.R) | Clase 4 |
| 07 | Outliers | [Outliers: Z-Score y Mahalanobis](temas/07-outliers-estadisticos/README.md) | [py](temas/07-outliers-estadisticos/codigo.py) | [R](temas/07-outliers-estadisticos/codigo.R) | Clase 5 |
| 08 | Outliers | [LOF e Isolation Forest](temas/08-lof-isolation-forest/README.md) | [py](temas/08-lof-isolation-forest/codigo.py) | [R](temas/08-lof-isolation-forest/codigo.R) | Clase 6 |
| 09 | Reducción de dimensiones | [PCA (componentes principales)](temas/09-pca/README.md) | [py](temas/09-pca/codigo.py) | [R](temas/09-pca/codigo.R) | Clase 6, Clase 7, Clase 8 |
| 10 | Reducción de dimensiones | [Multidimensional Scaling (MDS)](temas/10-mds/README.md) | [py](temas/10-mds/codigo.py) | [R](temas/10-mds/codigo.R) | Clase 7, Clase 8 |
| 11 | Reducción de dimensiones | [Análisis factorial](temas/11-analisis-factorial/README.md) | [py](temas/11-analisis-factorial/codigo.py) | [R](temas/11-analisis-factorial/codigo.R) | Clase 7, Clase 8, Clase 9 |
| 12 | Reducción de dimensiones | [Kernel PCA](temas/12-kernel-pca/README.md) | [py](temas/12-kernel-pca/codigo.py) | [R](temas/12-kernel-pca/codigo.R) | Clase 9 |
| 13 | Reducción de dimensiones | [SVD (descomposición en valores singulares)](temas/13-svd/README.md) | [py](temas/13-svd/codigo.py) | [R](temas/13-svd/codigo.R) | Clase 9 |
| 14 | Reducción de dimensiones | [Manifold learning: Isomap, t-SNE, LLE](temas/14-manifold/README.md) | [py](temas/14-manifold/codigo.py) | [R](temas/14-manifold/codigo.R) | Clase 8, Clase 9 |
| 15 | Supervisado | [EPE, MSE y el balance sesgo varianza](temas/15-sesgo-varianza/README.md) | [py](temas/15-sesgo-varianza/codigo.py) | [R](temas/15-sesgo-varianza/codigo.R) | Clase 10, Clase 11 |
| 16 | Supervisado | [Kernel regression y k vecinos](temas/16-kernel-regression/README.md) | [py](temas/16-kernel-regression/codigo.py) | [R](temas/16-kernel-regression/codigo.R) | Clase 10, Clase 11 |
| 17 | Supervisado | [Validación cruzada](temas/17-validacion-cruzada/README.md) | [py](temas/17-validacion-cruzada/codigo.py) | [R](temas/17-validacion-cruzada/codigo.R) | Clase 11, Clase 12 |
| 18 | Supervisado | [Selección de variables](temas/18-seleccion-variables/README.md) | [py](temas/18-seleccion-variables/codigo.py) | [R](temas/18-seleccion-variables/codigo.R) | Clase 12, Clase 13 |
| 19 | Supervisado | [Ridge, LASSO y Elastic Net](temas/19-ridge-lasso-elasticnet/README.md) | [py](temas/19-ridge-lasso-elasticnet/codigo.py) | [R](temas/19-ridge-lasso-elasticnet/codigo.R) | Clase 14 |

## Referencia rápida

Ver **[`cheatsheet.md`](cheatsheet.md)** para la tabla de funciones equivalentes entre R y Python.

## Sobre el origen del código

Cada bloque está marcado con su procedencia:

- **Del curso**: transcrito de los notebooks o de los archivos de código del profesor, sin cambios
  de fondo. Se corrigieron algunas erratas de los comentarios en español.
- **Equivalente agregado**: código que no estaba en el material del curso y se agregó para tener la
  versión del tema en el otro lenguaje o para completar un procedimiento. Verifícalo antes de
  usarlo como respuesta de un punto.

En el curso, la parte de clustering, outliers, manifold learning y todo lo supervisado se dictó en
**Python**. PCA, MDS, análisis factorial, Kernel PCA, SVD, selección de variables y penalización se
dictaron en **R**.

## Datos

La carpeta `datos/` tiene los archivos que usan los códigos en R: `measure.txt`, `digits.txt` y
`sponge.txt`.
