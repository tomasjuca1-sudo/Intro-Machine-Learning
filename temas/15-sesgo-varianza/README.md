# 15. EPE, MSE y el balance sesgo varianza

**Parte:** Supervisado  
**Clases:** Clase 10, Clase 11  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### El objetivo

Encontrar una función f que, dado X, prediga Y lo mejor posible. La función óptima bajo error
cuadrático es la media condicional:

```
f_optima(x) = E[ Y | X = x ]
```

Los datos se generan como:

```
Y = f_optima(X) + epsilon,   E[epsilon] = 0,  Var(epsilon) = sigma^2
```

### EPE (Expected Prediction Error)

Para un punto nuevo x0:

```
EPE(x0) = E[ (Y - fhat(x0))^2 ]
        = sigma^2  +  [ Sesgo(fhat(x0)) ]^2  +  Var( fhat(x0) )
```

Los tres términos:

1. **sigma^2**, error irreducible. No depende del modelo. Es el piso.
2. **Sesgo al cuadrado**, cuánto se aleja en promedio el modelo de la función real. Baja cuando sube
   la flexibilidad.
3. **Varianza**, cuánto cambia el modelo estimado si se cambia la muestra de entrenamiento. Sube
   cuando sube la flexibilidad.

El balance entre 2 y 3 produce la **curva en U** del error de test.

### MSE como estimador del EPE

```
                1
MSE = ---------------- * sum_i ( yhat_i - y_i )^2
          n_muestra
```

Regla central del curso, y la que más se pregunta:

- **MSE en train siempre baja** cuando el modelo se vuelve más flexible. Con flexibilidad máxima
  llega a cero. Por eso **no sirve para calibrar**.
- **MSE en test tiene forma de U**. El mínimo de esa curva es el nivel de flexibilidad calibrado.

### Flexibilidad

Cada modelo tiene su parámetro de flexibilidad:

| Modelo | Parámetro | Más flexible cuando |
|---|---|---|
| k vecinos | k | k pequeño |
| Kernel regression | h (ventana) | h pequeño |
| Regresión lineal | número de variables | más variables |
| Ridge, LASSO, Elastic Net | lambda | lambda pequeño |

Ojo con la dirección: en k vecinos y en kernel regression el parámetro es **inversamente**
proporcional a la flexibilidad.

### Por qué importa el tamaño de muestra

Con n pequeño, dos muestras de la misma población producen modelos muy distintos. Esa es la varianza
del estimador, y es la razón por la cual no se puede evaluar un modelo en los mismos datos con que
se entrenó.

---

## Cómo se relaciona el código con el tema

El notebook `class10_book1_intro` construye todo el escenario controlado:

- `f(x)` es la `f_optima`. Conocerla es un lujo que solo existe en simulación, y es lo que permite
  comparar la estimación contra la verdad.
- `sd = 1.2` es sigma, el error irreducible. Ese valor es el piso del MSE que ningún modelo baja.
- Generar `x1, y1` y `x2, y2` con `n = 10` y graficarlas lado a lado es la demostración visual de la
  varianza.

El notebook `class10_book2_kernel` es donde aparece la curva en U. El código compara al tiempo:

- `mse_train[i]`: calculado prediciendo sobre `x_train` con el modelo entrenado en `x_train`.
- `mse_test[i]`: calculado prediciendo sobre `x_test` con el modelo entrenado en `x_train`.

Al graficar ambas contra `hh`, la roja (train) es monótona y la azul (test) es convexa. El código
extrae el óptimo con `h_opt = hh[np.argmin(mse_test)]`.

Nota sobre la dirección del eje: como h grande significa menos flexibilidad, la curva se lee al
revés de la gráfica típica de flexibilidad. El comentario del propio código lo advierte.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | Escenario controlado: f_optima, sigma y muestras | Del curso, class10_book1_intro.ipynb |
| Python | Partición train y test manual | Del curso, class10_book2_kernel.ipynb |
| Python | Curva en U: MSE de train contra MSE de test | Del curso, class10_book2_kernel.ipynb |
| R | Curva en U en R con k vecinos | Equivalente agregado, no viene del material del curso |
