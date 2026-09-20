# ==========================================================================
# TEMA 10: MULTIDIMENSIONAL SCALING (MDS)
# Clase 7, Clase 8
# ==========================================================================


# --------------------------------------------------------------------------
# MDS clásico, métrico y no métrico (código del curso)
# Origen: Del curso, code3_class7_MDS.txt
# --------------------------------------------------------------------------

##################################
#MDS

data=eurodist

mds=cmdscale(data,k=2,eig=T)
mds

library(graphics)

x <- mds$points[, 1]
y <- -mds$points[, 2] # reflect so North is at the top

plot(x, y, type = "n", xlab = "", ylab = "", asp = 1, axes = FALSE,
     main = "cmdscale(eurodist)")
text(x, y, rownames(mds$points), cex = 0.6)

mds$eig


##################################
#MDS-metric

library(cluster)

#datos sobre esponjas marinas
  xx=read.table("sponge.txt",header=T,sep=",")

#cálculo de distancias tipo gower
  dis=daisy(xx,metric="gower")

mds=cmdscale(dis,k=2,eig=T)
mds

plot(mds$points)


##################################
#MDS non-metric

library(HSAUR2)
library(MASS)

data=voting

mds=isoMDS(voting)
mds

plot(mds$points)
plot(mds$points, type = "n", xlab = "", ylab = "", asp = 1, axes = FALSE,
     main = "MDS votes")
text(mds$points, rownames(mds$points), cex = 0.6)


# --------------------------------------------------------------------------
# Diagnóstico de calidad del MDS
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

mds <- cmdscale(eurodist, k = 2, eig = TRUE)

# Proporción de información retenida por los dos primeros ejes
sum(abs(mds$eig[1:2])) / sum(abs(mds$eig))

# Eigenvalores negativos grandes indican distancias no euclidianas
mds$eig[mds$eig < 0]

# Comparar distancias originales contra distancias en el mapa (diagrama de Shepard)
d_original <- as.vector(eurodist)
d_mapa <- as.vector(dist(mds$points))

plot(d_original, d_mapa, pch = 16, cex = 0.5,
     xlab = "Distancia original", ylab = "Distancia en el mapa")
abline(0, 1, col = "red")

cor(d_original, d_mapa)

# En el no métrico, el estrés está en el objeto
library(MASS)
mds_nm <- isoMDS(eurodist, k = 2)
mds_nm$stress   # en porcentaje. Menos de 5 es excelente, más de 20 es malo
