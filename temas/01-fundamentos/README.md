# 01. Axioma de los datos y tipos de aprendizaje

**Parte:** Fundamentos  
**Clases:** Clase 1, Clase 10  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### Idea central

Todo algoritmo de Machine Learning se apoya en el mismo supuesto estadístico: los datos son
**realizaciones independientes e idénticamente distribuidas** de un modelo de probabilidad
multivariado que describe el comportamiento poblacional que se quiere generalizar.

La factorización básica que usa el profesor es:

```
P_XY(x, y) = P_X(x) * P_Y(y | x)
```

### Las dos ramas del curso

**Aprendizaje supervisado.** Existe una variable de respuesta Y. El objetivo es encontrar una
función f que, dado X, prediga Y lo mejor posible. La parte P_Y(y|x) es la que interesa.
La función óptima es f_optima(x) = E[Y | X = x].

**Aprendizaje no supervisado.** No hay Y. El objetivo es encontrar patrones en la distribución
multivariada de X en R^p. La parte P_X(x) es la que interesa. De aquí salen los dos grandes
bloques del corte: clustering y reducción de dimensiones.

### Caso de datos por grupos

Cuando los datos vienen de grupos, la densidad se vuelve una mezcla:

```
f(x) = p1 * f1(x | clase 1) + p2 * f2(x | clase 2) + ...
```

Esta es exactamente la estructura que los algoritmos de clustering intentan recuperar sin
conocer las etiquetas. Es importante para el parcial: el clustering es un problema de estimar
una mezcla, no de clasificar.

### Descomposición del error (aparece en la parte supervisada)

```
Y = f_optima(X) + epsilon,   con E[epsilon] = 0 y Var(epsilon) = sigma^2
```

El término sigma^2 es el **error irreducible**. Ningún modelo puede bajar de ahí.

---

## Cómo se relaciona el código con el tema

El código de esta clase no estima nada, **simula**. Sirve para entender de donde vienen los datos.

- La función `f(x)` del notebook es la `f_optima`, es decir, la media condicional E[Y|X=x].
  Es la curva roja que se dibuja en todos los gráficos de la parte supervisada.
- `np.random.normal(0, sd, n)` genera el error irreducible epsilon. El parámetro `sd` controla
  sigma, que es el piso del MSE.
- El bloque que mezcla `m1` y `m2` con probabilidades `p1` y `p2` construye a mano la mezcla de
  densidades por grupos. Esos son los datos que después se le pasan a K-Means.
- Generar dos muestras pequeñas (`x1, y1` y `x2, y2`) y graficarlas lado a lado muestra la
  varianza del estimador: con n pequeño, dos muestras de la misma población se ven muy distintas.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | Simulación supervisada: f_optima más ruido | Del curso, class10_book1_intro.ipynb |
| Python | Efecto del tamaño de muestra en la varianza | Del curso, class10_book1_intro.ipynb |
| Python | Simulación no supervisada: normal multivariada y mezcla por grupos | Del curso, class1_book1_intro.ipynb |
| R | Equivalente en R de la simulación | Equivalente agregado, no viene del material del curso |
