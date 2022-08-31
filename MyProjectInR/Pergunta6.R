library(ggplot2)
set.seed(869)
four_vector <- vector(mode="numeric")
twentytwo_vector <- vector(mode="numeric")
eighty_vector <- vector(mode="numeric")
x <- vector(mode="numeric")

for (i in 1:290) {
  four_vector <- rbind(four_vector, runif(4, 6, 10))
  twentytwo_vector <- rbind(twentytwo_vector,runif(22, 6, 10))
  eighty_vector <- rbind(eighty_vector, runif(80, 6, 10))
}
four_average <- apply(four_vector, 1, mean)
twentytwo_average <- apply(twentytwo_vector, 1, mean)
eighty_average <- apply(eighty_vector, 1, mean)
four_expected_value <- mean(four_average)
twentytwo_expected_value <- mean(twentytwo_average)
eighty_expected_value <- mean(eighty_average)
four_sd <- sqrt(var(four_average)/4)
twentytwo_sd <- sqrt(var(twentytwo_average)/22)
eighty_sd <- sqrt(var(eighty_average)/80)
four_df <- data.frame(four_average)
twentytwo_df <- data.frame(twentytwo_average)
eighty_df <- data.frame(eighty_average)
ggplot(four_df, aes(x = four_average)) +
  geom_histogram(aes(y = ..density..), alpha = 0.3, binwidth = 0.05,
                 color = "red",) + theme(legend.position = "top") +
  theme(legend.position = "top") + theme(legend.position = "top") +
  geom_density() + 
  labs( y = "Frequência Relativa", x = "Média") + 
  ggtitle("Distribuição para n = 4")
ggplot(twentytwo_df, aes(x = twentytwo_average)) +
  geom_histogram(aes(y = ..density..), alpha = 0.3, binwidth = 0.05,
                 color = "red",) + theme(legend.position = "top") +
  theme(legend.position = "top") + theme(legend.position = "top") +
  geom_density() + 
  labs( y = "Frequência Relativa", x = "Média") + 
  ggtitle("Distribuição para n = 22")
ggplot(eighty_df, aes(x = eighty_average)) +
  geom_histogram(aes(y = ..density..), alpha = 0.3, binwidth = 0.05,
                 color = "red",) + theme(legend.position = "top") +
  theme(legend.position = "top") + theme(legend.position = "top") +
  geom_density() + 
  labs( y = "Frequência Relativa", x = "Média") + 
  ggtitle("Distribuição para n = 80")

''' myhist <- hist(four_average)
multiplier <- myhist$counts / myhist$density
mydensity <- density(four_average)
mydensity$y <- mydensity$y * multiplier[1]

plot(myhist)
lines(mydensity)

myhist <- hist(twentytwo_average)
multiplier <- myhist$counts / myhist$density
mydensity <- density(twentytwo_average)
mydensity$y <- mydensity$y * multiplier[1]

plot(myhist)
lines(mydensity)

myhist <- hist(eighty_average)
multiplier <- myhist$counts / myhist$density
mydensity <- density(eighty_average)
mydensity$y <- mydensity$y * multiplier[1]

plot(myhist)
lines(mydensity)'''
