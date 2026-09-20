# 19. Ridge, LASSO y Elastic Net

**Parte:** Supervisado  
**Clases:** Clase 14  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### El punto de partida

Mínimos cuadrados resuelve:

```
beta_hat = argmin  sum_i ( Y_i - beta^t X_i )^2
```

El problema: si hay **redundancia de información** entre las variables predictoras
(multicolinealidad), la estimación de los betas tiene mucha varianza y algunos betas toman valores
absolutos muy grandes. Los signos se vuelven inestables y el modelo predice mal fuera de muestra.

### La solución: penalizar

Se agrega un término que castiga el tamaño del vector de betas:

```
beta_hat = argmin  sum_i ( Y_i - beta^t X_i )^2  +  lambda * || beta ||
```

donde lambda > 0 es el **parámetro de penalización** (el parámetro de flexibilidad de estos
modelos). La diferencia entre los tres métodos es **cómo se define la norma**.

### Ridge, penalización cuadrática (norma 2)

```
beta_hat = argmin  sum_i ( Y_i - beta^t X_i )^2  +  lambda * sum_j beta_j^2
```

- **Encoge** los coeficientes hacia cero pero **nunca los hace exactamente cero**.
- Conserva todas las variables en el modelo.
- Funciona muy bien con multicolinealidad: reparte el peso entre variables correlacionadas.

### LASSO, penalización de valor absoluto (norma 1)

```
beta_hat = argmin  sum_i ( Y_i - beta^t X_i )^2  +  lambda * sum_j |beta_j|
```

- **Sí hace exactamente cero** algunos coeficientes. Es decir, **selecciona variables**
  automáticamente.
- Produce modelos dispersos (sparse) y más fáciles de interpretar.
- Entre variables muy correlacionadas tiende a escoger una y anular las demás, de forma algo
  arbitraria.

### Elastic Net, combinación de las dos

```
beta_hat = argmin  sum_i ( Y_i - beta^t X_i )^2  +  lambda * ( (1-alpha) sum_j beta_j^2 + alpha sum_j |beta_j| )
```

Dos parámetros de calibración: lambda y alpha.

- **alpha = 0**: es Ridge.
- **alpha = 1**: es LASSO.
- Intermedio: selecciona variables como LASSO pero maneja mejor los grupos de variables
  correlacionadas, tendiendo a incluirlas o excluirlas juntas.

### Cuidado con la convención de alpha

Esta es una trampa clásica de parcial. **La notación de la fórmula del curso y la de glmnet en R
están invertidas:**

| | alpha = 0 | alpha = 1 |
|---|---|---|
| Fórmula del curso | Ridge | LASSO |
| `glmnet` en R | Ridge | LASSO |
| `ElasticNet` en sklearn (`l1_ratio`) | Ridge | LASSO |

En la fórmula del curso alpha multiplica la norma 1, igual que en glmnet y que `l1_ratio` de
sklearn. Así que las tres coinciden. Lo que sí hay que recordar: en sklearn el parámetro de
penalización se llama **`alpha`** (es el lambda de la teoría) y la mezcla se llama **`l1_ratio`**.

### Escalamiento

Las tres penalizaciones dependen de la escala de las variables, porque penalizan el tamaño de los
coeficientes. **Siempre se estandariza antes.** `glmnet` lo hace por defecto con
`standardize = TRUE`.

### Cómo se ve el efecto de lambda

El gráfico típico pone lambda en el eje horizontal (en escala logarítmica y muchas veces invertido)
y los coeficientes en el vertical. Se ve como todos convergen a cero cuando lambda crece. En LASSO
los coeficientes tocan cero y se quedan ahí, en Ridge solo se acercan asintóticamente.

---

## Cómo se relaciona el código con el tema

### En Python

Cada modelo tiene **dos versiones** en sklearn:

| Sin CV | Con CV | Qué hace la versión CV |
|---|---|---|
| `Ridge(alpha=a)` | `RidgeCV(alphas=...)` | prueba toda la rejilla y se queda con la mejor |
| `Lasso(alpha=a)` | `LassoCV(alphas=...)` | igual |
| `ElasticNet(alpha=a)` | `ElasticNetCV(alphas=..., cv=5)` | igual |

La rejilla se arma siempre igual:

```python
n_alphas = 200
alphasCalibrar = np.logspace(-10, 2, n_alphas)   # de 10^-10 a 10^2, en escala log
```

Se usa escala logarítmica porque el efecto de lambda es multiplicativo, no aditivo.

Después de ajustar, `modeloRidge.alpha_` (con guion bajo al final) guarda el **lambda escogido**.
No confundir con el argumento `alphas` de entrada.

El bloque que gráfica los coeficientes usa `Ridge` sin CV dentro de un loop, porque necesita el
vector de betas para **cada** valor de la rejilla, no solo el óptimo:

```python
coefs = []
for a in alphasCalibrar:
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(XTrain, yTrain)
    coefs.append(ridge.coef_)
```

El `ax.set_xlim(ax.get_xlim()[::-1])` invierte el eje para que la flexibilidad crezca hacia la
derecha.

Nota: `store_cv_values=True` en `RidgeCV` cambio de nombre en versiones recientes de sklearn a
`store_cv_results`. Si da error, se quita ese argumento.

### En R

Una sola función, `glmnet`, y el tipo de penalización se controla con **`alpha`**:

```
cvmod = cv.glmnet(X, y, alpha=0)     # Ridge
cvmod = cv.glmnet(X, y, alpha=1)     # LASSO
cvmod = cv.glmnet(X, y, alpha=0.3)   # Elastic Net
```

Flujo estándar, que se repite idéntico para los tres:

1. `X = model.matrix(Grad.Rate~., train)[,-1]` convierte el data frame en matriz numérica.
   El `[,-1]` quita la columna del intercepto, que glmnet agrega por su cuenta. **Este paso es
   obligatorio**, glmnet no acepta fórmulas ni factores directamente.
2. `cvmod = cv.glmnet(X, y, alpha=...)` hace la validación cruzada sobre lambda.
3. `cvmod$lambda.min` es el lambda que minimiza el error de CV.
   `cvmod$lambda.1se` es el mayor lambda que queda dentro de un error estándar del mínimo, un modelo
   más simple y más robusto.
4. `plot(cvmod)` gráfica la curva de CV con las barras de error.
5. `glmnet(X, y, alpha=..., lambda=cvmod$lambda.min)` ajusta el modelo final.
6. `coef(mod_pen1)` muestra los coeficientes. **En LASSO los ceros aparecen como puntos.**
7. `predict(mod, Xtest)` y luego el MSE a mano.

Para el gráfico de las trayectorias de los betas se ajusta **sin** especificar lambda, y glmnet
recorre toda la secuencia: `plot(glmnet(X,y,alpha=1), xvar="lambda")`.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | Ridge con calibración por CV | Del curso, class14_book1_penalty.ipynb |
| Python | LASSO | Del curso, class14_book1_penalty.ipynb |
| Python | Elastic Net y comparación final | Del curso, class14_book1_penalty.ipynb |
| Python | Calibrar lambda y l1_ratio al tiempo | Equivalente agregado, no viene del material del curso |
| R | Ridge, LASSO y Elastic Net con glmnet (código del curso) | Del curso, code2_class14.txt |
| R | Detalles de glmnet que se piden en parciales | Equivalente agregado, no viene del material del curso |
