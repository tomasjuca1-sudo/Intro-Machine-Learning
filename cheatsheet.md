# Referencia rápida: funciones equivalentes

## Clustering

| Tarea | Python | R |
|---|---|---|
| K-Means | `KMeans(n_clusters=K, n_init=10).fit(X)` | `kmeans(X, centers=K, nstart=25)` |
| Centroides | `modelo.cluster_centers_` | `km$centers` |
| Etiquetas | `modelo.labels_` | `km$cluster` |
| WSS o SSE | `modelo.inertia_` | `km$tot.withinss` |
| Escalar | `scale(X)` de `sklearn.preprocessing` | `scale(X)` |
| K-Medoides | `kmedoids.fasterpam(D, K)` | `pam(X, k=K)` |
| Distancia de Gower | `gower.gower_matrix(X)` | `daisy(X, metric="gower")` |
| Matriz de distancias | `distance_matrix(X, X)` | `dist(X)` |
| Jerárquico, etiquetas | `AgglomerativeClustering(n_clusters=K, linkage=...)` | `cutree(hclust(d, method=...), k=K)` |
| Jerárquico, árbol | `linkage(X, method=...)` de scipy | `hclust(dist(X), method=...)` |
| Dendrograma | `dendrogram(arbol)` | `plot(hc)` |
| Cortar el árbol | `fcluster(arbol, t=K, criterion='maxclust')` | `cutree(hc, k=K)` |
| DBSCAN | `DBSCAN(eps=..., min_samples=...).fit(X)` | `dbscan(X, eps=..., minPts=...)` |
| Etiqueta de ruido | `-1` | `0` |
| Distancia al k vecino | `NearestNeighbors(n_neighbors=k)` | `kNNdistplot(X, k=k)` |

## Métricas de clustering

| Métrica | Python | R | Dirección |
|---|---|---|---|
| Silhouette | `silhouette_score(X, etiquetas)` | `mean(silhouette(cl, dist(X))[,3])` | máximo |
| Silhouette por punto | `silhouette_samples(X, etiquetas)` | `silhouette(cl, d)[,3]` | |
| Calinski-Harabasz | `calinski_harabasz_score(X, etiquetas)` | `cluster.stats(d, cl)$ch` | máximo |
| Davies-Bouldin | `davies_bouldin_score(X, etiquetas)` | `index.DB(X, cl)$DB` | mínimo |

## Detección de outliers

| Método | Python | R |
|---|---|---|
| Z-Score | `stats.zscore(x)` | `scale(x)` |
| Mahalanobis | `distance.mahalanobis(x, mu, inv_S)` | `mahalanobis(X, mu, S)` devuelve d^2 |
| Umbral chi cuadrado | `chi2.ppf(0.975, df=p)` | `qchisq(0.975, df=p)` |
| LOF | `LocalOutlierFactor(n_neighbors=k)` | `lof(X, minPts=k)` |
| Score de LOF | `negative_outlier_factor_` (negativo es atípico) | `lof()` directo (mayor a 1 es atípico) |
| Isolation Forest | `IsolationForest(n_estimators=50)` | `isolation.forest()` de isotree |
| Predicción | `-1` outlier, `1` inlier | según el paquete |

## Reducción de dimensiones

| Método | Python | R |
|---|---|---|
| PCA | `PCA().fit_transform(X)` | `princomp(X, scores=T)` o `prcomp(X, scale.=T)` |
| Eigenvalores | `pca.explained_variance_` | `pca$sdev^2` |
| Varianza explicada | `pca.explained_variance_ratio_` | `summary(pca)` |
| Cargas | `pca.components_` (fila por componente) | `pca$loadings` o `pr$rotation` (columna por componente) |
| Scores | `pca.transform(X)` | `pca$scores` o `pr$x` |
| Descomposición espectral | `np.linalg.eigh(S)` | `eigen(S)` |
| Biplot | a mano | `biplot(pca)` |
| MDS clásico | `MDS(dissimilarity='precomputed')` | `cmdscale(D, k=2, eig=T)` |
| MDS no métrico | `MDS(metric=False)` | `isoMDS(D)` de MASS |
| Análisis factorial | `FactorAnalyzer(n_factors=d, rotation='varimax')` | `factanal(X, d)` |
| Unicidades | `fa.get_uniquenesses()` | `fa$uniquenesses` |
| Kernel PCA | `KernelPCA(kernel='rbf', gamma=...)` | `kpca(~., data, kernel="rbfdot", kpar=list(sigma=...))` |
| SVD | `np.linalg.svd(X)` | `svd(X)` |
| Isomap | `manifold.Isomap(n_neighbors=k)` | `isomap(dist(X), k=k)` de vegan |
| t-SNE | `manifold.TSNE(perplexity=...)` | `Rtsne(X, perplexity=...)` |
| LLE | `manifold.LocallyLinearEmbedding(method=...)` | `lle(X, m=2, k=k)` |

## Supervisado

| Tarea | Python | R |
|---|---|---|
| Partición train y test | `train_test_split(X, y, test_size=0.33, random_state=0)` | `sample()` más indexado |
| K vecinos | `KNeighborsRegressor(n_neighbors=k)` | `knn.reg(train, test, y, k)` de FNN |
| Regresión lineal | `LinearRegression().fit(X, y)` | `lm(y ~ ., data)` |
| MSE | `np.average(np.square(pred - y))` | `mean((y - pred)^2)` |
| KFold | `KFold(n_splits=10)` | `sample(rep(1:K, length.out=n))` |
| Validación cruzada automática | `cross_val_score(modelo, X, y, cv=10, scoring="neg_mean_squared_error")` | `cv.glm(datos, modelo, K=10)$delta[1]` |
| Búsqueda de parámetro | `GridSearchCV(modelo, rejilla, cv=10)` | `train()` de caret |
| Forward | `SequentialFeatureSelector(direction='forward')` | `regsubsets(..., method="forward")` |
| Exhaustivo | `ExhaustiveFeatureSelector()` de mlxtend | `regsubsets(..., method="exhaustive")` |
| Cp, BIC, R2 ajustado | a mano o con statsmodels | `summary(subs)$cp`, `$bic`, `$adjr2` |
| AIC y BIC | `modelo.aic`, `modelo.bic` de statsmodels | `AIC(lm1)`, `BIC(lm1)` |
| Ridge | `RidgeCV(alphas=...)` | `cv.glmnet(X, y, alpha=0)` |
| LASSO | `LassoCV(alphas=...)` | `cv.glmnet(X, y, alpha=1)` |
| Elastic Net | `ElasticNetCV(alphas=..., l1_ratio=...)` | `cv.glmnet(X, y, alpha=0.3)` |
| Lambda escogido | `modelo.alpha_` | `cvmod$lambda.min` |
| Coeficientes | `modelo.coef_` | `coef(modelo)` |
| Matriz de diseño | ya es matriz | `model.matrix(y ~ ., datos)[,-1]` |

## Cosas que se olvidan en el parcial

1. **Escalar antes** de K-Means, jerárquico, DBSCAN, PCA y cualquier penalización.
2. En DBSCAN el ruido es `-1` en Python y `0` en R.
3. `negative_outlier_factor_` de LOF está **negado**: mientras más negativo, más atípico.
4. `cross_val_score` devuelve el MSE con **signo negativo** (`neg_mean_squared_error`).
5. `glmnet` no acepta data frames: hay que pasar por `model.matrix(...)[,-1]`.
6. En `regsubsets` hay que correr `na.omit()` antes.
7. `pca.components_` de sklearn tiene una **fila** por componente y `pr$rotation` de R tiene una
   **columna** por componente. Están transpuestas entre sí.
8. En k vecinos y en kernel regression el parámetro es **inversamente** proporcional a la
   flexibilidad: k grande o h grande significa un modelo menos flexible.
9. El MSE de **train** no sirve para calibrar, porque siempre baja con la flexibilidad.
10. `mahalanobis()` de R ya devuelve la distancia **al cuadrado**.
