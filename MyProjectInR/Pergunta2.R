library(ggplot2)
library(reshape2)
library(readxl)

os_meus_dados = read_excel("Desktop/EsperancaVida.xlsx",
                     sheet = "Quadro",
                     range = "A7:CY70",
                     na = "0")

dados <- os_meus_dados[c(45:62), c(1,47,52,54,81,86,88)]

colnames(dados) <- c("Anos",
                    "SI - Eslovénia - Homens",
                    "IE - Irlanda - Homens",
                    "GR - Grécia - Homens",
                    "SI - Eslovénia - Mulheres",
                    "IE - Irlanda - Mulheres",
                    "GR - Grécia - Mulheres")

dados_arranjados <- melt(dados, id = "Anos")

dados_arranjados$value <- round(as.numeric(dados_arranjados$value), digit = 1)

colnames(dados_arranjados) <- c('Tempo(anos)',
                        'País/Sexo',
                        'Esperança de vida à nascença')

ggplot(dados_arranjados, aes(x = `Tempo(anos)`,
                     y = `Esperança de vida à nascença`,
                     col = `País/Sexo`)) + geom_line()


