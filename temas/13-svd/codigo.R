# ==========================================================================
# TEMA 13: SVD (DESCOMPOSICIÓN EN VALORES SINGULARES)
# Clase 9
# ==========================================================================


# --------------------------------------------------------------------------
# SVD e information retrieval (código del curso)
# Origen: Del curso, code3_cass9_SVD.txt
# --------------------------------------------------------------------------

##########################################################################
#Spectral Value Decomposition
#Information Retrieval Application

#Library for text mining
  library(tm)

#data or 20 articles about crude oil
  data("crude")

#create the term-document frequency matrix
  tdm <- TermDocumentMatrix(crude, control = list(removePunctuation = TRUE,
                          stopwords = TRUE))

#Spectral value decomposition
  tt=as.matrix(tdm)
  dim(tt)

  ss=svd(tt)

  ss$d #check for u and v

  #check the decomposition
  tt-ss$u%*%diag(ss$d)%*%t(ss$v)

#Aproximation to lower dimensión (k=10)
  ttk=ss$u[,1:10]%*%diag(ss$d[1:10])%*%t(ss$v)[1:10,]

  ttk3=ss$u%*%diag(c(ss$d[1:10],rep(0,10)))%*%t(ss$v)

  ttk-ttk3

#Matrices in reduced rank
  k=20

  uu=ss$u[,1:k]
  dd=ss$d[1:k]
  v=ss$v[,1:k]

#Search - query
  set.seed(2189)
  q=runif(1000,0,1)
  q=ifelse(q>=0.5,1,0)

  qq=diag(dd)%*%t(uu)%*%q

#Cosine calculations:
  cos=v%*%qq/sqrt(diag(t(v)%*%v)*rep(t(qq)%*%qq,k))


# --------------------------------------------------------------------------
# SVD básico y relación con PCA
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

X <- as.matrix(iris[, 1:4])

# SVD sobre datos centrados
Xc <- scale(X, center = TRUE, scale = FALSE)
ss <- svd(Xc)

ss$d        # valores singulares
ss$u        # vectores izquierdos
ss$v        # vectores derechos

# Relación con PCA: v son las cargas y u %*% diag(d) son los scores
pr <- prcomp(X, center = TRUE, scale. = FALSE)

round(abs(ss$v) - abs(pr$rotation), 10)          # iguales salvo signo
round(abs(ss$u %*% diag(ss$d)) - abs(pr$x), 10)  # iguales salvo signo

# Eigenvalores a partir de valores singulares
n <- nrow(X)
ss$d^2/(n-1)
pr$sdev^2

# Aproximación de rango r
r <- 2
Xr <- ss$u[, 1:r] %*% diag(ss$d[1:r]) %*% t(ss$v[, 1:r])

norm(Xc - Xr, type = "F")    # error de Frobenius
sum(ss$d[1:r]^2)/sum(ss$d^2) # proporción retenida
