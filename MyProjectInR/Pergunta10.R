library(tidyverse)
library(ggplot2)

sc <- vector(mode = "numeric")
cc <- vector(mode = "numeric")
non_contaminated <- vector(mode = "numeric")
contaminated <- vector(mode = "numeric")
nc_ic_aver <- vector(mode = "numeric")
cc_ic_aver <- vector(mode = "numeric")
samples <- vector(mode = "numeric")

l = 2500
m <- 850
n <- 100
lambda <- 0.73
lambdaf <- 0.01
seed <- 915

m_ic <- function(average){
  return((2*(qnorm((1 + 0.95)/2)/sqrt(n)))/average)
}

while (n <= l){
  set.seed(seed)
  sc <- replicate(m, rexp(n, lambda))
  average_nc <- apply(sc, 2,mean)
  m_ic_nc <- sapply(average_nc, m_ic)
  nc_ic_aver <- mean(m_ic_nc)
  non_contaminated <- c(non_contaminated, nc_ic_aver)
  cc <- sc
  cc[1:(n/4),] <- replicate(m, rexp(n/4, lambdaf))
  average_cc <- apply(cc, 2,mean)
  m_ic_cc <- sapply(average_cc, m_ic) 
  cc_ic_aver <- mean(m_ic_cc)
  contaminated <- c(contaminated, cc_ic_aver)
  samples <- c(samples,n)
  n <- n + 100
}
df <- data.frame(non_contaminated, contaminated, samples)

ggplot() + 
  geom_point(data = df, mapping = aes(x = samples, y = non_contaminated, colour = "Não Contaminado"), size = 1) +
  geom_point(data = df, mapping = aes(x = samples, y = contaminated, colour = "Contaminado"), size = 1) + 
  labs(x = "Amostras", y = "MAc(blue), MA(red)", colour = "Legenda") +
  ggtitle("Média de Amplitudes Contaminada e Não Contaminada")



  







