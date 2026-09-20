# ==========================================================================
# TEMA 11: ANÁLISIS FACTORIAL
# Clase 7, Clase 8, Clase 9
# ==========================================================================


# --------------------------------------------------------------------------
# factanal: iris y Holzinger (código del curso)
# Origen: Del curso, code4_class7_factorial.txt y code1_class9_factorial.txt
# --------------------------------------------------------------------------

#########################################################################
##Ejemplo Sencillo con un factor
##Datos de las flores de Iris (en paquete datasets)

##Carga de Paquetes
  library(datasets)
  library(MVA)

##Carga de Datos
  x=iris
  xx=x[,1:4]  #Usando las primeras 4 columnas

##Análisis factorial con datos iris
##factanal es el procedimiento con argumentos (datos, número de factores)
  fa=factanal(xx,1)
  fa  ##Acá se ven los resultados del modelo estimado

##Se puede hacer con solo la matriz de covarianza
  S=var(xx)
  fa2=factanal(covmat=S,n.obs=150,factors=1)

  fa3=factanal(xx,2)  ##Pruebe que con dos factores no se puede

##Análisis factorial calculando los scores (valores del factor por observación)
  fa=factanal(xx,1,scores="regression")
  scor=fa$scores  ##Contiene el vector con el factor de cada flor (tamaño)

##Visualizando los factores
  plot(scor)  ##Dispersión del único factor (no muy interesante)
  plot(density(scor))  ##Densidad del factor (distribución: más útil)

##Visualizando los factores por familia de flores (las tres de iris)
  s1=scor[1:50]
  s2=scor[51:100]
  s3=scor[101:150]

  plot(density(s1),xlim=c(-1.5,2),lwd=2)
  lines(density(s2),lwd=2,col=2)
  lines(density(s3),lwd=2,col=3)


#########################################################################
##Real Example
##Holzinger ability test data (9 measures)

  library(psych)

##Cargando datos (matriz de correlaciones)
  SS=Holzinger.9
  cor.plot(SS)  ##Para visualizar la matriz de correlaciones

##Determinando el número de factores a usar
  factanal(factors=1,covmat=SS,n.obs=145)  ##Mirar el p-value de la prueba chi
  factanal(factors=2,covmat=SS,n.obs=145)
  factanal(factors=3,covmat=SS,n.obs=145)  ##Este parece ser...

##Análisis Factorial con tres factores
  ff=factanal(factors=3,covmat=SS,n.obs=145)
  ff  ##Interprete los factores obtenidos (con rotación varimax)

##Cambiando la rotación de los factores (promax)
  ff2=factanal(factors=3,covmat=SS,n.obs=145,rotation="promax")
  ff2  ##Interprete los factores obtenidos


# --------------------------------------------------------------------------
# Lectura de la salida de factanal
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

ff <- factanal(factors = 3, covmat = Holzinger.9, n.obs = 145)

ff$loadings      # cargas. Las que no se imprimen son menores a 0.1
ff$uniquenesses  # unicidad de cada variable
1 - ff$uniquenesses   # comunalidad

ff$STATISTIC     # estadístico chi cuadrado
ff$dof           # grados de libertad
ff$PVAL          # p-value: si es grande, d factores son suficientes

# Mostrar todas las cargas sin cortar
print(ff$loadings, cutoff = 0)

# Suma de cuadrados de cargas por factor (varianza explicada)
colSums(ff$loadings^2)


# Alternativa del paquete psych, más flexible en rotaciones y métodos
library(psych)
fa_psych <- fa(r = Holzinger.9, nfactors = 3, n.obs = 145,
               rotate = "varimax", fm = "ml")
fa_psych
fa.diagram(fa_psych)   # diagrama de qué variable carga en qué factor

# Gráfico de sedimentación para decidir el número de factores
scree(Holzinger.9)
fa.parallel(Holzinger.9, n.obs = 145)
