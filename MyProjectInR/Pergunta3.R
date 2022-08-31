library(readxl)
library(ggplot2)
library(reshape2)

meus_dados <- read_xlsx("Desktop/QualidadeARO3.xlsx")

antas_espinho <- as.numeric(unlist(c(meus_dados[1])))
paio_pires <- as.numeric(unlist(c(meus_dados[7])))

dados <- data.frame(variable = c(rep("antas_espinho", length(antas_espinho)),
                                 rep("paio_pires", length(paio_pires))),
                    value = c(antas_espinho, paio_pires))

my_plot <- ggplot(dados, aes(x = value, fill = variable, colour = variable)) +
  geom_histogram(bins = 100, color = "black", alpha = 0.6, bondary = 0, position = "identity") +
  theme(plot.title = element_text(hjust = 0.7)) + 
  scale_x_continuous(breaks = seq(0, 204, 4)) + scale_y_continuous(breaks = seq(0, 1200, 50))

print(my_plot + labs(title = "Histograma dos Níveis de ozono na estação Paio_Pires e Antas-Espinhos",
                     y = "Número de Contagens", x = expression("Níveis de O3"), fill = "Estações"))


