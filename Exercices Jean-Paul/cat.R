price <- lm(data$price~., data = data) 
library(coefplot)
coefplot(price)
scatter.smooth(data$price, data$sqft)

plot(price)
meanprice = mean(data$price)
meanprice

cor.price.sqft <- cor(data$sqft,data$price)
plot(cor.price.sqft*data$price)

scatter.smooth(cor.price.sqft)
boxplot(data$price,data$sqft)
table(data$sqft)

library(dplyr)
cat1 <- filter(data, sqft<1000) 
cat2 <- filter(data, sqft>=1000 & sqft<1999)
cat3 <- filter(data, sqft>=2000 & sqft<2999)
cat4 <- filter(data, sqft>=3000)

mean(cat1,cat2,cat3,cat4)

cat = c(cat1, cat2, cat3, cat4)
sqftprice1<-c(cat1$sqft/cat1$price)
hist(cat$sqft, mean)
boxplot(cat1$sqft, cat2$sqft, cat3$sqft, cat4$sqft)
hist(sqftprice1)

library("FactoMineR")
library("factoextra")

barplot(cat1$sqft, cat1$price, xlab="Prix", ylab="Sqft", main = "Prix par rapport Sqft")
plot(cat1$sqft, cat1$price, xlab="Sqft", ylab="Prix", main = "Prix par rapport Sqft")

par(mfrow=c(2,2))
scatcat1<- plot(cat1$sqft, cat1$price, xlab="Sqft", ylab="Prix", main = "Prix par rapport Sqft (Cat1)", col="blue")
scatcat2<- plot(cat2$sqft, cat2$price, xlab="Sqft", ylab="Prix", main = "Prix par rapport Sqft (Cat2)",col="red")
scatcat3<- plot(cat3$sqft, cat3$price, xlab="Sqft", ylab="Prix", main = "Prix par rapport Sqft (Cat3)",col= "orange")
scatcat4<- plot(cat4$sqft, cat4$price, xlab="Sqft", ylab="Prix", main = "Prix par rapport Sqft (Cat4)",col="green")


cat1meanprice<-mean(cat1$price) 
cat2meanprice<-mean(cat2$price)
cat3meanprice<-mean(cat3$price)
cat4meanprice<-mean(cat4$price)
AllMean<-c(cat1meanprice+cat2meanprice+cat3meanprice+cat4meanprice)/4
AllMean

sumcat2<-lm(cat2$price~cat2$baths+cat2$sqft+cat2$beds)
sumcat2
reg.multiple <- step(sumcat2, direction = "backward")
library(coefplot)
coefplot(reg.multiple)
library(hrbrthemes)
library(ggplot2)

p2 <- ggplot(cat3, aes(x=price, y=sqft)) +
  geom_point() +
  geom_smooth(method=lm , color="red", fill="#69b3a2", se=TRUE)+
  theme_ipsum()
p2
