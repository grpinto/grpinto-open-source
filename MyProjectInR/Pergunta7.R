set.seed(1095)

BVector <- vector(mode = "numeric")

for (i in 1:140) {
  BVector <- rbind(BVector, rbinom(56, 15, 0.73))
}

BAverage <- apply(BVector, 1, mean)

TAverage <- mean(BAverage)

abs(TAverage - 15*0.73)

