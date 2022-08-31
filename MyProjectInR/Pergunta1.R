library(readxl)
library(plyr)
library(tidyverse)
library(ggplot2)

os_meus_dados <- read_excel("Desktop/ResiduosPerCapita.xlsx", 
                      sheet = "Quadro", 
                      range ="A12:C43")

dados_essenciais <- os_meus_dados[c(5,15,24),]

dados <- dados_essenciais %>% gather(key = Ano, 
                            value = value, 
                            `2004`: `┴ 2018`)

colnames(dados) <- c('Grupos/Países', 
                        'Ano', 
                        'Resíduos per capita(tonelada)')

dados[dados == "┴ 2018"] <- "2018"

ggplot(dados, aes(Ano, `Resíduos per capita(tonelada)`, 
                     fill = `Grupos/Países`)) + geom_col(position = "dodge") +
  ggtitle("Resíduos per capita em toneladas da Grécia, Polónia e 
                        Bulgária no ano de 2004 e 2018")





