# ==========================================================================
# TEMA 19: RIDGE, LASSO Y ELASTIC NET
# Clase 14
# ==========================================================================


# --------------------------------------------------------------------------
# Ridge, LASSO y Elastic Net con glmnet (código del curso)
# Origen: Del curso, code2_class14.txt
# --------------------------------------------------------------------------

library(ISLR)
college=College
college=college[,-1]

set.seed(23)
n1=sample(seq(1:dim(college)[1]),dim(college)[1]-200)
train=college[n1,]
test=college[-n1,]

########################################################################
#2. Penalización cuadrática (Ridge)

library(glmnet)

#glmnet necesita matrices, no data frames. El [,-1] quita el intercepto
X=model.matrix(Grad.Rate~.,train)[,-1]
Xtest=model.matrix(Grad.Rate~.,test)[,-1]

y=train$Grad.Rate
ytest=test$Grad.Rate

cvmod=cv.glmnet(X,y,alpha=0)
cvmod$lambda.min
plot(cvmod)

#Gráfico de betas
  mod_pen2_plot=glmnet(X,y,alpha=0)
  plot(mod_pen2_plot,xvar=c("lambda"))

mod_pen2=glmnet(X,y,alpha=0,lambda=cvmod$lambda.min)
coef(mod_pen2)

predp2=predict(mod_pen2,Xtest)
msep2=mean((ytest-predp2)^2)
msep2


########################################################################
#3. Penalización norma 1 (Lasso)

cvmod=cv.glmnet(X,y,alpha=1)
cvmod$lambda.min
plot(cvmod)

  mod_pen1_plot=glmnet(X,y,alpha=1)
  plot(mod_pen1_plot,xvar=c("lambda"))

mod_pen1=glmnet(X,y,alpha=1,lambda=cvmod$lambda.min)
coef(mod_pen1)   #los coeficientes en cero aparecen como punto

predp1=predict(mod_pen1,Xtest)
msep1=mean((ytest-predp1)^2)
msep1


########################################################################
#4. Elastic Net

cvmod=cv.glmnet(X,y,alpha=0.3)
cvmod$lambda.min
plot(cvmod)

  mod_pen_en_plot=glmnet(X,y,alpha=0.3)
  plot(mod_pen_en_plot,xvar=c("lambda"))

mod_pen_en=glmnet(X,y,alpha=0.3,lambda=cvmod$lambda.min)
coef(mod_pen_en)

predp_en=predict(mod_pen_en,Xtest)
msep_en=mean((ytest-predp_en)^2)
msep_en


########################################################################
#5. Comparación modelos

mse       #mínimos cuadrados con selección de variables
msep2     #Ridge
msep1     #LASSO
msep_en   #Elastic Net


# --------------------------------------------------------------------------
# Detalles de glmnet que se piden en parciales
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

library(glmnet)

# lambda.min contra lambda.1se
cvmod <- cv.glmnet(X, y, alpha = 1)

cvmod$lambda.min   # minimiza el error de CV
cvmod$lambda.1se   # el mayor lambda a un error estándar del mínimo (modelo más simple)

coef(cvmod, s = "lambda.min")
coef(cvmod, s = "lambda.1se")

# Número de variables distintas de cero para cada lambda
cvmod$nzero

# Ver qué variables sobreviven en LASSO
betas <- coef(cvmod, s = "lambda.min")
betas[betas[, 1] != 0, , drop = FALSE]


# Calibrar alpha y lambda al tiempo
alphas <- seq(0, 1, by = 0.1)
resultados <- data.frame(alpha = alphas, lambda = NA, mse_cv = NA)

set.seed(1)
foldid <- sample(1:10, size = length(y), replace = TRUE)   # mismos folds para comparar

for (i in seq_along(alphas)) {
  cv_i <- cv.glmnet(X, y, alpha = alphas[i], foldid = foldid)
  resultados$lambda[i] <- cv_i$lambda.min
  resultados$mse_cv[i] <- min(cv_i$cvm)
}

resultados
mejor <- resultados[which.min(resultados$mse_cv), ]
mejor

# Modelo final
modelo_final <- glmnet(X, y, alpha = mejor$alpha, lambda = mejor$lambda)
coef(modelo_final)


# Trayectorias de coeficientes
mod_path <- glmnet(X, y, alpha = 1)
plot(mod_path, xvar = "lambda", label = TRUE)
plot(mod_path, xvar = "dev", label = TRUE)     # contra desviación explicada
plot(mod_path, xvar = "norm", label = TRUE)    # contra norma L1


# glmnet estandariza por defecto. Para desactivarlo:
# glmnet(X, y, alpha = 1, standardize = FALSE)
