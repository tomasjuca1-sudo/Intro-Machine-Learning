# 14. Manifold learning: Isomap, t-SNE, LLE

**Parte:** Reducción de dimensiones  
**Clases:** Clase 8, Clase 9  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### Qué es un manifold

Un manifold es una superficie de dimensión baja embebida en un espacio de dimensión alta. El ejemplo
del curso es la **S-curve**: una lámina de 2 dimensiones doblada dentro de R^3. Los datos viven en
3D, pero su estructura real es 2D.

PCA falla aquí porque solo aplana linealmente: dos puntos en extremos opuestos del doblez quedan
cerca en la proyección, aunque sobre la superficie estén lejísimos.

### ISOMAP

Se puede entender como una **extensión de MDS**. La idea es definir las distancias como la **ruta
más corta** entre dos puntos, siempre y cuando la ruta se haga a través del manifold. Es decir, para
llegar de un punto a otro hay que pasar por vecindades pobladas, un concepto parecido al de
densidades en DBSCAN.

Son dos pasos:

1. **Ruta más corta**: se construye un grafo conectando cada punto con sus k vecinos más cercanos y
   se calculan las distancias geodésicas con el **algoritmo de Dijkstra**.
2. **MDS**: con esas nuevas distancias se construyen coordenadas nuevas usando MDS.

Parámetro clave: `n_neighbors`. Si es muy grande, el grafo hace atajos por fuera del manifold
(short-circuit) y se pierde la estructura. Si es muy pequeño, el grafo se desconecta.

### t-SNE (T-distributed Stochastic Neighbor Embedding)

Convierte distancias en **probabilidades de vecindad**: la probabilidad de que i escoja a j como
vecino. Luego busca coordenadas en 2D cuyas probabilidades se parezcan, minimizando la divergencia
de Kullback-Leibler. Usa una distribución t de Student en el espacio destino, lo que da colas
pesadas y evita que todo se apiñe en el centro.

Parámetro clave: **perplexity**, que es aproximadamente el número efectivo de vecinos. Valores
típicos entre 5 y 50.

Advertencias importantes para el parcial: t-SNE preserva bien la **estructura local** pero **no la
global**. Las distancias entre clusters en un gráfico de t-SNE no son interpretables, y los tamaños
de los clusters tampoco. Además no es determinístico y no proyecta datos nuevos.

### LLE (Local Linear Embedding)

Supone que cada punto se puede escribir como **combinación lineal de sus vecinos**. Estima esos
pesos W_ij en el espacio original y luego busca coordenadas de dimensión baja que conserven esos
mismos pesos. La idea es que localmente el manifold es plano.

Variantes que aparecen en el código:

| Método | Idea |
|---|---|
| `standard` | LLE clásico |
| `ltsa` | Local Tangent Space Alignment, alinea espacios tangentes locales |
| `hessian` | Hessian eigenmap, usa la curvatura, requiere más vecinos |
| `modified` | Modified LLE, usa varios vectores de pesos para estabilizar |

### Comparación rápida

| Método | Preserva | Datos nuevos | Determinístico |
|---|---|---|---|
| PCA | varianza global | sí | sí |
| Isomap | distancias geodésicas globales | sí | sí |
| LLE | geometría local lineal | sí | depende |
| t-SNE | vecindades locales | no | no |

---

## Cómo se relaciona el código con el tema

Todo sale del módulo `manifold` de sklearn, y los tres métodos comparten la misma interfaz:
`fit_transform(S_points)`.

**Datos.** `datasets.make_s_curve(n_samples, random_state=0)` devuelve dos cosas: `S_points`, que son
las coordenadas en 3D, y `S_color`, que es la posición de cada punto **a lo largo de la curva**. Ese
color es el criterio de evaluación: si el embedding es bueno, el degradado de color queda continuo y
ordenado en 2D. Si el color se mezcla, el método rompió el manifold.

**Las funciones de graficación** (`plot_3d`, `plot_2d`, `add_2d_scatter`) son utilitarias del
notebook, no parte del método. Vale la pena tenerlas a mano porque el parcial puede pedir el gráfico.

**Isomap:** `manifold.Isomap(n_neighbors=10, n_components=2, p=1)`. El `p=1` fija la métrica de
Minkowski en distancia Manhattan.

**t-SNE:** `manifold.TSNE(n_components=2, perplexity=10, init="random", max_iter=250,
random_state=0)`. Con `max_iter=250` el algoritmo apenas converge, es un valor bajo para que corra
rápido en clase.

**LLE:** el notebook usa un truco útil. Define un diccionario `params` con los argumentos comunes y
lo desempaqueta con `**params` en las cuatro variantes:

```
params = {"n_neighbors": n_neighbors, "n_components": n_components,
          "eigen_solver": "auto", "random_state": 0}
lle_standard = manifold.LocallyLinearEmbedding(method="standard", **params)
```

Así solo cambia el argumento `method` entre `standard`, `ltsa`, `hessian` y `modified`.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | Datos S-curve y funciones de graficación | Del curso, class9_book1_manifold.ipynb |
| Python | Isomap y t-SNE | Del curso, class9_book1_manifold.ipynb |
| Python | Las cuatro variantes de LLE | Del curso, class9_book1_manifold.ipynb |
| R | Isomap, LLE y t-SNE en R | Equivalente agregado, no viene del material del curso |
