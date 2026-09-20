# ==========================================================================
# TEMA 12: KERNEL PCA
# Clase 9
# ==========================================================================


# --------------------------------------------------------------------------
# kpca con kernlab (código del curso)
# Origen: Del curso, code2_class9_KPCA.txt
# --------------------------------------------------------------------------

library(kernlab)
# another example using the iris

  data(iris)
  test <- sample(1:150,70)
  kpc <- kpca(~.,data=iris[-test,-5],kernel="rbfdot",
              kpar=list(sigma=0.2),features=2, eta=0.001, maxiter=65)
#print the principal component vectors
  pcv(kpc)
#plot the data projection on the components
  plot(predict(kpc,iris[,-5]),col=as.integer(iris[,5]),
       xlab="1st Principal Component",ylab="2nd Principal Component")

#embed remaining points
  emb <- predict(kpc,iris[test,-5])
  points(emb,col=as.integer(iris[test,5]))


#################################
#circular data for KPCA
library(plotly)
packageVersion('plotly')

  y1=runif(80,-1.5,1.5)
  x1=runif(80,-1.5,1.5)
  z1=sqrt(rep(7,80)-(y1*y1+rnorm(80,0,.01))-(x1*x1+rnorm(80,0,.01)))

  y2=runif(80,-0.6,1)
  x2=runif(80,-.6,1)
  z2=sqrt(rep(3,80)-(y2*y2+rnorm(80,0,.01))-(x2*x2+rnorm(80,0,.01)))

  dat1=as.data.frame(cbind(x1,y1,z1))
  dat1$clas=rep(1,80)
  dat1$clas=as.factor(dat1$clas)
  plot_ly(dat1,x=~x1,y=~y1,z=~sqrt(z1))

  dat2=as.data.frame(cbind(x2,y2,z2))
  dat2$clas=rep(2,80)
  dat2$clas=as.factor(dat2$clas)
  plot_ly(dat2,x=~x2,y=~y2,z=~sqrt(z2))

  colnames(dat1)=c("x","y","z","clas")
  colnames(dat2)=c("x","y","z","clas")
  dat=as.data.frame(rbind(dat1,dat2))

  plot_ly(dat,x=~x,y=~y,z=~sqrt(z),color=~clas, colors = c('#BF382A', '#0C4B8E'))


##################################
#Projection on PCA
  kk=princomp(dat[,1:3],scores=T)
  plot(kk$scores[,1:2],col=dat$clas)


##################################
#Projection on KPCA
kpc <- kpca(~.,data=dat[,1:3],kernel="rbfdot",
       kpar=list(sigma=1.1),features=2, eta=0.001, maxiter=65)
  plot(pcv(kpc),col=dat$clas)

plot(predict(kpc,dat),col=as.integer(dat$clas),
xlab="1st Principal Component",ylab="2nd Principal Component")


########################################################################
#Unos datos conocidos
 data=read.table("measure.txt")
 head(data)
  X=data[,1:3]
  pca=princomp(X,scores=T)
  pca$scores
  par(mfrow=c(1,2))
  plot(pca$scores[,1],pca$scores[,2],col=c(2,4)[data[,4]])

#Ahora con kernel
  kpc= kpca(~.,X,kernel="rbfdot",
         kpar=list(sigma=0.1),features=2, eta=0.001, maxiter=65)
pcv(kpc)
plot(predict(kpc,X),col=c(2,4)[data[,4]],
xlab="1st Principal Component",ylab="2nd Principal Component")
