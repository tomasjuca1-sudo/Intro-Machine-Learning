# ==========================================================================
# TEMA 09: PCA (COMPONENTES PRINCIPALES)
# Clase 6, Clase 7, Clase 8
# ==========================================================================


# --------------------------------------------------------------------------
# PCA paso a paso y con princomp (código del curso)
# Origen: Del curso, code1_class7_ejemplo1-pca.txt
# --------------------------------------------------------------------------

data=read.table("measure.txt")

X=data[,1:2]
n=dim(X)[1]
S=var(X)

#Spectral decomposition (finding eigenvalues and eigenvectors)
  eigen(S)

#Solución a primer componente principal
  w=eigen(S)$vectors[,1]
  lambda=eigen(S)$values[1]

#New variable: First Component
  medias=cbind(rep(colMeans(X)[1],n),rep(colMeans(X)[2],n))
  XX=as.matrix(X-medias)  #datos centrados
  Z=XX%*%w
  plot(density(Z))

#Points reconstruction
  ZZ=cbind(Z*w[1],Z*w[2])
  ZZ=ZZ+medias  #agregar la medias otra vez
  plot(ZZ)
  plot(X,lwd=2)
  points(ZZ,col=2,lwd=2)

#Using princomp
  pp2=princomp(X)
  summary(pp2,loadings=T)
  sqrt(lambda)
  lambda/sum(eigen(S)$values)


#De tres a dos dimensiones
  X=data[,1:3]
  pca=princomp(X,scores=T)
  summary(pca,loadings=T)
  pca$scores
  par(mfrow=c(1,2))
  biplot(pca)
  plot(pca$scores[,1],pca$scores[,2],col=c(2,4)[data[,4]])

#Gráfico de codo
  plot(pca$sdev^2,type="b",main="Varianzas de Componentes")


#Ahora cambiamos de base de datos
data=iris

  X=data[,1:4]
  pca=princomp(X,scores=T)
  summary(pca,loadings=T)
  pca$scores
  par(mfrow=c(1,2))
  biplot(pca)
  plot(pca$scores[,1],pca$scores[,2],col=c(2,4,3)[data[,5]])


# --------------------------------------------------------------------------
# PCA sobre dígitos manuscritos y reconstrucción de imágenes
# Origen: Del curso, code2_class7_ejemplo2-pca.txt
# --------------------------------------------------------------------------

#######################################################
#data: handwritten digits

  data=read.csv("digits.txt",head=F)
  pix=data[,1:64]
  y=data[,65]

##########################################
#PLOT
index=211

pixi=matrix(rep(0,64),ncol=8)
for(i in 1:8){
  for(j in 1:8){
    pixi[j,i]=pix[index,(i-1)*8+j]
  }
}
x=seq(0,1,by=1/8)
yy=seq(0,1,by=1/8)
image(x, yy, pixi,col = terrain.colors(100))
y[index]

##########################################
#Principal components
S=var(pix)
library(corrplot)
M=cor(pix)
corrplot(M, method="circle")

pp=princomp(pix,scores=T)
summary(pp)

#Gráfico de codo
  plot(pp$sdev^2,type="b",main="Varianzas de Componentes")

##############################################
#Images reconstruction
nn=dim(pix)[1]
p=dim(pix)[2]

  w=eigen(S)$vectors

    medias = colMeans(pix)
    pixx=as.matrix(pix - rep(medias, rep.int(nrow(pix), ncol(pix))))
    Z=pixx%*%w

#Projections
#number of components
  k=6
  ZZ=Z%*%rbind(t(w)[1:k,],matrix(rep(0,(p-k)*p),ncol=p))
  ZZ=as.matrix(ZZ+rep(medias, rep.int(nrow(pix), ncol(pix))))

index=18

pixi=matrix(rep(0,64),ncol=8)
pixi2=matrix(rep(0,64),ncol=8)

for(i in 1:8){
  for(j in 1:8){
    pixi[j,i]=pix[index,(i-1)*8+j]
    pixi2[j,i]=ZZ[index,(i-1)*8+j]
  }
}

x=seq(0,1,by=1/8)
yy=seq(0,1,by=1/8)

par(mfrow=c(2,1))
image(x, yy, pixi,col = terrain.colors(100))
image(x, yy, pixi2,col = terrain.colors(100))

y[index]


# --------------------------------------------------------------------------
# prcomp: la alternativa a princomp
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

# prcomp usa SVD y es numéricamente más estable que princomp
X <- iris[, 1:4]

pr <- prcomp(X, center = TRUE, scale. = TRUE)   # scale. = TRUE usa la correlación

pr$sdev^2        # eigenvalores
pr$rotation      # cargas (una COLUMNA por componente)
pr$x             # scores

summary(pr)      # varianza explicada y acumulada

plot(pr$sdev^2, type = "b", main = "Gráfico de codo")
biplot(pr)

# Varianza acumulada
cumsum(pr$sdev^2)/sum(pr$sdev^2)

# Regla de Kaiser sobre la correlación: eigenvalores mayores a 1
which(pr$sdev^2 > 1)
