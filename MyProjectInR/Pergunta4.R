library("readxl")
library(ggplot2)

my_data <- read_excel("Desktop/Utentes.xlsx", sheet = "Sheet1")

my_u <- my_data[,c("Idade", "IMC")]

ggplot(my_u, aes(x = Idade, y = IMC)) + geom_point(color = "darkblue", size = 3) +
  scale_y_continuous(breaks = seq(50, 110, by = 5)) + 
  scale_x_continuous(breaks = seq(30, 80, by = 5)) + 
  ylab("Indice de massa corporal") + 
  geom_smooth(method = "lm", formula = y ~ x, se = F, color = "black") + 
  ggtitle("                       Gráfico de dispersão que relaciona a 
                       idade com o indice de massa corporal")
