## Exercice 1

# Creer un vecteur RT
rt = c(4, 7.2, 4.6, 8.3)
# Moyenne RT
mean(rt)
# Ecart-type rt
sd(rt)
# Creer logrt
rtlog<-log(rt)
log(rt)
# Créer un replicate d'Id
Id <-c("ID1", "ID2", " ID3", "ID4")
rep(Id, each=4)

## Exercice 2
# Afficher summary Data
summary(mens)
#Afficher les 4 dernière lignes
tail(mens, 4)
#Afficher les 10 premiere lignes
head(mens, 10)
# Calculer Quantile
quantile(mens$biceps)
# Challenge : Calculer l’erreur standard de la moyenne de la colonne biceps.
err.std <- sd(mens$biceps)/sqrt(length(mens))
err.std
# Challenge : la fonction tapply(vecteur, facteur, fonction) 
chevilleNIVEAU<-tapply(mens$cheville, mens$NIVEAU, sd)
print(chevilleNIVEAU)
# Challenge ++ : Calculer la moyenne de cheville pour chaque combinaison niveau d’entraînement/genre.
tapply(mens$cheville, mens$NIVEAU)
tapply(mens$cheville, list(mens$NIVEAU, mens$GENRE), mean)


## Exercice 3

#	Installer et charger le package dplyr
library(dplyr)
cpn<-mens[,c("cheville","poignet","NIVEAU")]
# Table sans col cuisse
menCui<-mens[,-6]
# Classer tableau ordre décroissant

mens[order(mens$NIVEAU),]
mens[order(-mens$cheville),]

# Log Cheville - Ajouter col
logCheville = log(mens$cheville)
logCheville
mens2<-mutate(mens,logCheville)
#Filtrer Femme Tr1
filter(mens, GENRE=="femme"& NIVEAU=="tr1")
filter(mens, GENRE=="femme"& cheville <20)
filter(mens, NIVEAU!="tr3")

## Exercice 4

library(psychotools)
