# 11. Análisis factorial

**Parte:** Reducción de dimensiones  
**Clases:** Clase 7, Clase 8, Clase 9  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### En qué se diferencia de PCA

PCA es una **transformación algebraica**: los componentes son combinaciones lineales de las
variables observadas. No hay modelo estadístico detrás.

El análisis factorial es un **modelo de variables latentes**: se postula que existen factores no
observados que generan las variables observadas.

```
X = Lambda F + epsilon
```

- `F`: los factores comunes (las variables latentes), de dimensión d.
- `Lambda`: la matriz de **cargas** (loadings), de dimensión p por d.
- `epsilon`: las unicidades, la parte específica de cada variable que ningún factor explica.

De ahí sale la descomposición de la covarianza:

```
S = Lambda Lambda^T + Psi
```

Para cada variable:

```
Var(X_j) = comunalidad_j + unicidad_j
```

La **comunalidad** es la proporción de varianza de la variable explicada por los factores comunes.
La **unicidad** (uniqueness) es lo que queda. En R, `factanal` reporta directamente las unicidades.

### Ejemplos de variables latentes del curso

1. ¿Cómo se mide la inteligencia de las personas? El factor es la habilidad, las variables son los
   puntajes de varias pruebas.
2. Unmixing, el problema de la fiesta de cóctel: separar voces a partir de micrófonos.
3. En iris, el factor único se interpreta como el **tamaño** de la flor.

### Número de factores

`factanal` entrega una **prueba chi cuadrado** de bondad de ajuste. La hipótesis nula es que d
factores son suficientes.

- **p-value pequeño**: se rechaza, hacen falta más factores.
- **p-value grande**: d factores bastan.

Se va subiendo d hasta que el p-value deje de ser significativo. En el ejemplo de Holzinger el
resultado son **tres factores**.

Con iris pasa algo instructivo: con un factor funciona, con dos el modelo **no se puede estimar**
porque no hay suficientes grados de libertad (4 variables no alcanzan para 2 factores).

### Rotación

Las cargas no son únicas: cualquier rotación ortogonal de Lambda da el mismo ajuste. La rotación se
escoge para facilitar la **interpretación**, buscando que cada variable cargue fuerte en un solo
factor.

- **varimax** (la de por defecto): rotación ortogonal, los factores quedan no correlacionados.
- **promax**: rotación oblicua, permite que los factores estén correlacionados. Suele ser más
  realista cuando las variables latentes tienen relación entre si.

### Scores

Son los valores estimados de los factores para cada observación. Se piden con
`scores = "regression"`. Son el análogo de `pca$scores`.

---

## Cómo se relaciona el código con el tema

Todo el código del curso está en **R** y gira alrededor de una sola función: `factanal`.

Firma básica: `factanal(datos, numero_de_factores)`.

Puntos que el código demuestra a propósito:

1. **Se puede estimar solo con la matriz de covarianza o de correlación**, sin los datos crudos:
   `factanal(covmat = S, n.obs = 150, factors = 1)`. Esto es distinto a la mayoría de métodos y es
   útil porque muchos estudios publican solo la matriz de correlaciones. El ejemplo de Holzinger
   trabaja exactamente así, con `SS = Holzinger.9`.

2. **`fa3 = factanal(xx, 2)` falla a propósito.** El comentario del profesor dice "Pruebe que con
   dos factores no se puede". Con p = 4 variables no hay grados de libertad para 2 factores.

3. **Los scores hay que pedirlos:** `factanal(xx, 1, scores = "regression")` y luego `fa$scores`.
   El código gráfica `plot(density(scor))` porque con un solo factor el diagrama de dispersión no
   dice nada, la densidad sí.

4. **Interpretación por grupos:** se parte el vector de scores en los tres subconjuntos de 50 flores
   (`s1 = scor[1:50]`, etc.) y se superponen las tres densidades. Ahí se ve que el factor separa las
   especies aunque nunca se le dio esa información.

5. **Comparación de rotaciones:** se corre `factanal(factors=3, covmat=SS, n.obs=145)` con varimax
   por defecto y luego con `rotation = "promax"`, y se comparan las interpretaciones.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | Análisis factorial con factor_analyzer | Equivalente agregado. El código del curso está en R |
| Python | FactorAnalysis de sklearn | Equivalente agregado, no viene del material del curso |
| R | factanal: iris y Holzinger (código del curso) | Del curso, code4_class7_factorial.txt y code1_class9_factorial.txt |
| R | Lectura de la salida de factanal | Equivalente agregado, no viene del material del curso |
