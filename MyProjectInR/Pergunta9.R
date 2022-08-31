library(tidyverse)
library(ggplot2)

ic_average <- vector(mode = "numeric")
samples <- vector(mode = "numeric")

n <- 100

m_ic <- function(seed, m, n, lambda, gama){
  set.seed(seed)
  return(mean(replicate(m, (2*(qnorm((1 + gama)/2)/
                                 sqrt(n)))/(mean(rexp(n, lambda))))))
}

while (n <= 5000) {
  ic_average <- c(ic_average, m_ic(seed = 728, 
                                   m = 1450, n, lambda = 0.83, gama = 0.9))
  samples <- c(samples, n)
  n <- n + 100
}

df <- data.frame(ic_average,samples)

dfplot <-ggplot(df, aes_(samples,ic_average)) + 
  geom_point( size = 1, colour = "red")

print(dfplot + labs(y = "MA(n)", x = "Dimensão da Amostra") + 
        ggtitle("Intervalo de confiança consoante a dimensão da amostra"))



