### MODELO DE REGRESSÃO LOGÍSTICA MULTINOMIAL PARA A DISSERTAÇÃO DE MESTRADO

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")

setwd("D:/15.1 Mestrado/1.0 - Dissertação de mestrado/2. python_mestrado")     # diretório onde está arquivo

library(readxl)
data <- read_excel("base_regressao_logistica3.xlsx")
data
summary(data)


#library(gmodels)
#CrossTable(data$Ano, data$Valor_Pago)

### Codificando as 

ano <- c("anoum", "anodois", "anotres", "anoquatro", "anocinco", "anoseis")
ano_cod <- as.numeric(factor(data$Ano, levels = ano))

mes <- c("janeiro","fevereiro","marco","abril","maio","junho","julho","agosto","setembro","outubro","novembro","dezembro")
mes_cod <- as.numeric(factor(data$Mês, levels = mes))

comarca_cod <- as.numeric(factor(data$Comarca))   # quando não coloca condição, vai por ordem alfabética

uf <- c("ap","to","rr","al","ac","pa","se","ms","pi","ma","rn","ce","pb","am","es","sc","go","rs","pe","mt","df","ro","pr","mg","ba","rj","sp")
uf_cod <- as.numeric(factor(data$UF))

valor <- c("baixo","baixomedio","medio","medioalto","alto")
dmoral_cod <- as.numeric(factor(data$`Dano Moral`, levels = valor))
dmaterial_cod <- as.numeric(factor(data$`Dano Material`, levels = valor))
valor_cod <- as.numeric(factor(data$Valor_Pago, levels = valor))

andamento <- c("procedente", "procedenteemparte", "improcedente", "outro")
andamento_cod <- as.numeric(factor(data$ANDAMENTO_PROCESSO, levels = andamento))

### Criando dummies das colunas Motivo e Causa

#install.packages("fastDummies")
library(fastDummies)

data_dummies <- dummy_cols(data,
                           select_columns = c("MOTIVO PROCESSO", "CAUSA_PROCESSO"))     # transforma essas colunas em dummies
data_dummies <- data_dummies[, -c(1,2,3,4,5,6,7,8,9,10)]    # deixa data_dummies só com as dummies: todas as linhas e exclui as 10 primeiras colunas, que não são dummies
str(data_dummies)
#data_dummies$`MOTIVO PROCESSO_overbooking` <- NULL # excluir porque causa multicolinearidade (quando rodo no STATA fala isso)
#data_dummies$CAUSA_PROCESSO_covid <- NULL # excluir porque causa multicolinearidade (quando rodo no STATA fala isso)
#data_dummies$CAUSA_PROCESSO_trafegoaereo <- NULL # excluir porque causa multicolinearidade (quando rodo no STATA fala isso)

### Juntando colunas codificadas com as dummy

data_cod <- data.frame(ano_cod,mes_cod,comarca_cod,uf_cod,
                       dmoral_cod,dmaterial_cod,andamento_cod,valor_cod,data_dummies)
#View(data_cod)

### Tranformar o atributo classe (valor pago) em fator

data_cod$valor_cod <- as.factor(data_cod$valor_cod)
str(data_cod)



### MOdeling
### Multinomial Logistic Regression
#install.packages("nnet")
library(nnet)

data_cod$valor_cod <- relevel(data_cod$valor_cod, ref = "1")        # valor de referência é o da classe 1: baixo
data$Ano <- as.factor(data$Ano)
mymodel <- multinom(valor_cod ~., data = data_cod)               # valor_cod ~. : valor_cod é a dependente e ~. significa que todas as coutras colunass são as independtens
summary(mymodel)      # mostra os coeficientes para os dados de treino. Esses coeficientes são os logaritmos das razões de chances

razao_de_chance <- exp(summary(mymodel)$coefficients)                  # esse mostra as razões de chances, ou seja, quanto maior de chance uma classe pode acontecer em relação á classe de referência


mymodel_oim <- multinom(valor_cod ~ 1, data = data_cod) # modelo sem regressores, so com interceto (only intercept model)

### 2-tailed z-test

z <- summary(mymodel)$coefficients/summary(mymodel)$standard.errors
p <- (1 - pnorm(abs(z), 0, 1))*2                                                # p-valores

### Confusion matriX and misclassification error training dataset

pred <- predict(mymodel, data_cod)       # predições para os dados de treino
tab <- table(pred, data_cod$valor_cod)   # confusion matrix for training data set

accuracy <- sum(diag(tab)/sum(tab)) # acurácia
misclassif <- 1 - accuracy          # missclassification error

### Predicition and model assessment
n <- table(data_cod$valor_cod)
freq_classes <- n/sum(n)                                      # frequency of classes

tab/colSums(tab)                              # acurácia para cada classe

anova(mymodel_oim, mymodel)                   # comparar modelo só com intercepto e modelo completo
# p-value <5% indica que o modelo como um todo se adequa significativamente melhor do que o modelo so com intercepto


### Goodness of fit
# check the predicted probability for each category
pred_proba <- head(mymodel$fitted.values, 30)
#predicted result for each class
predicted <- head(predict(mymodel),30)
# test the goodness of fit
quiqua <- chisq.test(data_cod$valor_cod, predict(mymodel))      

#frequências esperadas: pressuposto: frequencias esperadas >5 
quiqua$expected

# residual padroizado ajkistado: > 1,96 ou < -1,96. Se estiver nessas regiões, os residuos são estaticamente significante
quiqua$stdres


### Pseudo  R2
#install.packages("DescTools")
library(DescTools)
PseudoR2(mymodel, which = c("CoxSnell", "Nagelkerke", "McFadden"))


