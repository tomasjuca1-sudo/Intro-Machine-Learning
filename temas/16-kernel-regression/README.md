# 16. Kernel regression y k vecinos

**Parte:** Supervisado  
**Clases:** Clase 10, Clase 11  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### Kernel regression rectangular

Para predecir Y* dado un valor particular X*, se buscan en la muestra de entrenamiento los valores
observados X_i que caigan dentro del intervalo [X* - h, X* + h] alrededor de X*. La predicción es el
**promedio de los Y_i** correspondientes:

```
                sum_i  I( |X_i - X*| <= h ) * Y_i
fhat(X*) =  -------------------------------------
                sum_i  I( |X_i - X*| <= h )
```

donde I(a) es la función indicadora, que vale 1 cuando la condición es cierta y 0 cuando es falsa.

### Relación con k vecinos

Son primos. La diferencia:

- **k vecinos**: se fija el **número** de puntos cercanos. La ventana cambia de ancho según la
  densidad local.
- **Kernel regression**: se fija el **ancho** de la ventana. El número de puntos cambia según la
  densidad local.

En ambos la predicción es un promedio local de los Y.

### El parámetro h

h controla la flexibilidad de forma **inversa**:

- **h grande**: la ventana abarca muchos puntos, la función estimada queda muy suave. Hay **sesgo**,
  falta flexibilidad. En el ejemplo del curso, h = 0.4 ya suaviza demasiado.
- **h pequeño**: la ventana abarca pocos puntos, la función estimada sigue el ruido. Hay mucha
  **varianza**, exceso de flexibilidad. Con h = 0.03 la curva estimada es un zigzag.

La calibración no se puede hacer a ojo. Se recorre una secuencia de valores de h, se calcula el MSE
en la muestra de test para cada uno, y se toma el h que lo minimiza.

### Problema de las colas

Cerca de los extremos del rango de X la ventana queda incompleta y hay menos puntos. Ahí el
estimador es menos confiable. En el código esto se ve en el `max(..., 0.1)` del denominador, que
evita una división por cero cuando no cae ningún punto dentro de la ventana.

---

## Cómo se relaciona el código con el tema

La función `kernelreg(xstar, X, Y, h)` es el corazón del tema. Sus argumentos:

- `xstar`: los puntos **nuevos** donde se quiere predecir.
- `X`, `Y`: los datos de **entrenamiento**.
- `h`: el semiancho de la ventana.

Línea por línea:

```python
dist = abs(xstar[i] - X)                  # distancia del punto nuevo a cada dato de train
zz = np.where(dist <= h, Y, 0)            # deja el Y si está dentro de la ventana, si no pone 0
pred[i] = sum(zz) / max(sum(np.where(dist <= h, 1, 0)), 0.1)
```

- `sum(zz)` es el numerador de la fórmula: la suma de los Y dentro de la ventana.
- `sum(np.where(dist <= h, 1, 0))` es el denominador: cuántos puntos cayeron dentro.
- El `max(..., 0.1)` protege contra la división por cero cuando la ventana está vacía.

Fíjate en el orden de los argumentos al calcular los dos MSE, porque es el error más común:

```python
fhat_train = kernelreg(x_train, x_train, y_train, hh[i])   # predice EN train, entrenado CON train
fhat_test  = kernelreg(x_test,  x_train, y_train, hh[i])   # predice EN test,  entrenado CON train
```

El segundo argumento y el tercero **siempre son los datos de entrenamiento**. Lo único que cambia es
donde se predice.

Para k vecinos la versión con librería es `KNeighborsRegressor(n_neighbors=vecinos)` y el flujo es
el estándar de sklearn: `.fit(XTrain, yTrain)` y luego `.predict(XTest)`.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | Implementación de kernel regression | Del curso, class11_book1.ipynb |
| Python | Calibración de h y modelo final | Del curso, class11_book1.ipynb |
| Python | K vecinos con sklearn y MSE en test | Del curso, class11_book2_CV.ipynb |
| R | Kernel regression y k vecinos en R | Equivalente agregado, no viene del material del curso |
