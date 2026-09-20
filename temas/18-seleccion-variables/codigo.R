# ==========================================================================
# TEMA 18: SELECCIÓN DE VARIABLES
# Clase 12, Clase 13
# ==========================================================================


# --------------------------------------------------------------------------
# regsubsets: exhaustivo y forward (código del curso)
# Origen: Del curso, code1_class13.txt
# --------------------------------------------------------------------------

######################################################################
##Paquetes requeridos
  library(ISLR)
  library(leaps)

######################################################################
##Datos: atributos de jugadores de Baseball
  fix(Hitters)
#Para quitarle los datos faltantes
  Hitters=na.omit(Hitters)


######################################################################
##Modelo lineal usando todas las variables
  lm1=lm(Salary~.,data=Hitters)
  summary(lm1)
  AIC(lm1);  BIC(lm1)


######################################################################
##Búsqueda Exhaustiva
#La función es "regsubsets" del paquete "leaps"
#Se escribe el modelo como "Y~X"="Dependiente~independientes"
#El parámetro L (máximo de variables) se controla con "nvmax"

  reg_subset=regsubsets(Salary~.,Hitters,nvmax=12,method="exhaustive")

  reg_sub_summary=summary(reg_subset)
  reg_sub_summary

#Argumentos del resultado
  reg_sub_summary$cp  #(los Cp de Mallows para cada mejor modelo con cada k)

#Gráficos
  plot(reg_sub_summary$cp,type="b")               #Cp de Mallows
  plot(reg_sub_summary$bic,type="b",col="red")    #BIC
  plot(reg_sub_summary$adjr2,type="b",col="blue") #R2 ajustado


######################################################################
##Método Forward
##Se escribe igual, solo cambia el método:

  reg_subset=regsubsets(Salary~.,Hitters,method="forward",nvmax=19)


# --------------------------------------------------------------------------
# Extraer el mejor modelo y sus coeficientes
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

library(ISLR)
library(leaps)

Hitters <- na.omit(Hitters)

subs <- regsubsets(Salary ~ ., Hitters, nvmax = 19, method = "exhaustive")
res <- summary(subs)

# Mejor tamaño según cada criterio
which.min(res$cp)      # Cp de Mallows: mínimo
which.min(res$bic)     # BIC: mínimo
which.max(res$adjr2)   # R2 ajustado: máximo
which.min(res$rss)     # RSS: siempre el modelo más grande, por eso no sirve

# Coeficientes del mejor modelo de tamaño k
k <- which.min(res$bic)
coef(subs, k)

# Qué variables entraron
names(coef(subs, k))[-1]

# Gráficos de los tres criterios juntos
par(mfrow = c(1, 3))
plot(res$cp, type = "b", xlab = "Número de variables", ylab = "Cp")
points(which.min(res$cp), min(res$cp), col = "red", cex = 2, pch = 20)

plot(res$bic, type = "b", xlab = "Número de variables", ylab = "BIC")
points(which.min(res$bic), min(res$bic), col = "red", cex = 2, pch = 20)

plot(res$adjr2, type = "b", xlab = "Número de variables", ylab = "R2 ajustado")
points(which.max(res$adjr2), max(res$adjr2), col = "red", cex = 2, pch = 20)
par(mfrow = c(1, 1))

# Gráfico propio de regsubsets: qué variables entran en cada mejor modelo
plot(subs, scale = "bic")


# Comparar exhaustivo, forward y backward
exh  <- regsubsets(Salary ~ ., Hitters, nvmax = 19, method = "exhaustive")
fwd  <- regsubsets(Salary ~ ., Hitters, nvmax = 19, method = "forward")
bwd  <- regsubsets(Salary ~ ., Hitters, nvmax = 19, method = "backward")

coef(exh, 7)
coef(fwd, 7)
coef(bwd, 7)   # no tienen que coincidir


# Selección por AIC paso a paso con step()
modelo_completo <- lm(Salary ~ ., data = Hitters)
modelo_nulo <- lm(Salary ~ 1, data = Hitters)

step_fwd <- step(modelo_nulo,
                 scope = list(lower = modelo_nulo, upper = modelo_completo),
                 direction = "forward")

step_bwd <- step(modelo_completo, direction = "backward")

# Para usar BIC en vez de AIC se pasa k = log(n)
step_bic <- step(modelo_completo, direction = "backward", k = log(nrow(Hitters)))


# --------------------------------------------------------------------------
# Selección con MSE en test
# Origen: Del curso, code2_class14.txt (primera parte)
# --------------------------------------------------------------------------

library(ISLR)
college=College
college=college[,-1]

########################################################################
#Se parte la muestra en dos: train y test
#Se usa el MSE con la muestra test

set.seed(23)
n1=sample(seq(1:dim(college)[1]),dim(college)[1]-200)
train=college[n1,]
test=college[-n1,]


########################################################################
#Modelo con el mejor subconjunto de variables
library(leaps)

subs=regsubsets(Grad.Rate~.,train,nvmax=14)
sum_subs=summary(subs)
plot(sum_subs$cp)
which.min(sum_subs$cp)
sum_subs

#Best choice
lm1=lm(Grad.Rate~Apps+Enroll+Top25perc+F.Undergrad+P.Undergrad+
       Outstate+Room.Board+Personal+perc.alumni,data=train)
summary(lm1)
pred=predict(lm1,test)
mse=mean((test$Grad.Rate-pred)^2)
mse
