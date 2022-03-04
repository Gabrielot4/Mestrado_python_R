### MODELO DE REGRESSÃO LOGÍSTICA MULTINOMIAL PARA A DISSERTAÇÃO DE MESTRADO

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")

setwd("D:/15.1 Mestrado/1.0 - Dissertação de mestrado/2. python_mestrado")     # diretório onde está arquivo

library(readxl)
data <- read_excel("base_reduzido.xlsx")
data
summary(data)


#Excluir colunas que não interessam
data$Cia_dummy <- NULL
data$Mês <- NULL
data$Mês_ <- NULL
data$Ano <- NULL
data$Comarca <- NULL
data$UF <- NULL
data$Orgao <- NULL
data$`Dano Moral` <- NULL
data$`Dano Material` <- NULL
data$`Valor Pago` <- NULL

data



#Tranformando variáveis categóricas em fator e definindo as referências com base nas maiores frequências
data$Cia <- relevel(as.factor(data$Cia), ref = 2) # ref = latam
data$Mês__ <- relevel(as.factor(data$Mês__), ref=3) # ref = nf (meses de nao ferias)
data$Ano_ <- relevel(as.factor(data$Ano_), ref=2) # ref = 2020
data$Região <- relevel(as.factor(data$Região), ref=4)
data$Dano_Moral_red <- relevel(as.factor(data$Dano_Moral_red), ref=2)
data$Dano_Material_red <- relevel(as.factor(data$Dano_Material_red), ref=2)
data$motivo <- relevel(as.factor(data$motivo), ref=2)
data$causa <- relevel(as.factor(data$causa), ref=1)
data$andamento <- relevel(as.factor(data$andamento), ref=3)
data$Valor_pago_red <- relevel(as.factor(data$Valor_pago_red), ref=2)


### Modeling
### Multinomial Logistic Regression

#install.packages("nnet")
library(nnet)

# mymodel: modelo com todos os regressores
mymodel <- multinom(Valor_pago_red ~., data = data)               # valor_pago_red ~. : valor_pago_red é a dependente e ~. significa que todas as outras colunass são as independtens
summary(mymodel)      # mostra os coeficientes para os dados de treino. Esses coeficientes são os logaritmos das razões de chances

odds_ratio <- exp(summary(mymodel)$coefficients)  # razão de chances dos coeficientes estimandos: exp(ln(Betas))

# mymodel_oi: modelo só com intercepto
mymodel_oi <- multinom(Valor_pago_red ~1, data=data)
summary(mymodel_oi)


### 2-tailed z-test
z <- summary(mymodel)$coefficients/summary(mymodel)$standard.errors
p <- (1 - pnorm(abs(z), 0, 1))*2  

### Predicted probabilities 
head(pp.values <- mymodel$fitted.values) # Probabilidade de prever baix ou medio ou alto, em número, para cada linha do dataframe
pp.values
head(pp.outcome <- predict(mymodel, data), 6) # Categorias que foram preditas
pp.outcome

### Confusion Matrix before crossvalidation
conf_m <- table(pp.outcome, data$Valor_pago_red)
conf_m

### Accuracy before crossvalidation
acc <- sum(diag(conf_m))/sum(conf_m)
acc

### ROC Curve
install.packages("pROC")
library(pROC)
multiclass.roc(data$Valor_pago_red, mymodel$fitted.values)

### Likelihood-ratio test (chis statistic): para rejeitar ou não a hipótese de os coefs dos regressores serem iguais a zer
### H0: B1 = B2 = Bj = 0 | se p-value < 0.01, então rejeita-se HO
### Model fit information: 
anova(mymodel_oi, mymodel) # rejeita-se H0, ou seja, os regressores do modelo completo têm influência nas respostas da variável dependente


### Goodness of fit: dizer no texto do mestrado que o teste chi quadrado aqui é para goodness of fit, posi existe variso outros para outras coisas
### Testar se existe diferença estatisticamente significante entre os valores esperados e os observados
### H0: existe diferença entre os valores observados e esperados
### H1: não existe diferença
### p-value < alpha: rejeita HO, ou seja, não há evidências significantes de que existe diferenças entres os valores observados e esperados
chi2 <- chisq.test(data$Valor_pago_red, pp.outcome) # chisq.test(expected, observed)
chi2 # p-value < 0,01: rejeita H0


### Outra forma do Goodness of fit, mais fácil de compreender o resultado
install.packages("lsr")
library(lsr)

goodnessOfFitTest(pp.outcome, p = c(baixo=.3364, medio=.3438, alto=.3198)) # goodnessOfFitTest(observado, esperado) | os valores esperados são as proporções das categorias presentes na base de dados original

### Chi-square independence test, utiliza o mesmo chisq.text acima, mas as entras são as duas variáveis que ser analisar.
### Testar se há associação entre duas vairiáveis categóricas, ou seja, se são independentes ou não
# H0: não há associação entre as variáveis preditoras e a dependente
# H1: há associação entre as variáveis preditoras e a dependente
# p < 5%: rejeita H0, ou seja, há associação entre as variáveis


### Log Likelihood ratio test
### Analisar quais preditores permitem prever a categoria de resposta, ou seja, se os regressores fazem diferença no modelo
### HO: coeficiente Beta da variável x1 é zerp
library(lmtest)
lrtest(mymodel, mymodel_oi) # comparação entre modelo completo e só com intercepto. p<alpha rejeita H0,ou seja, os preditores são importantes no modelo

colunas = c('Cia', 'Mês__', 'Ano_', 'Região', "Dano_Moral_red", "Dano_Material_red", "motivo", "causa", "andamento" )

# significância de cada preditor no modelo
for (i in colunas) {
  lr <- lrtest(mymodel, i)  
  print(lr)
}



### Pseudo  R2
library(DescTools)
PseudoR2(mymodel, which = c("CoxSnell", "Nagelkerke", "McFadden"))

### CrossValidation with caret
library(caret)
fit.control <- trainControl(method = 'repeatedcv', number = 5, repeats = 10)
fit <- train(Valor_pago_red ~., data=data, method = "multinom", trControl = fit.control, trace=FALSE)
fit
confusionMatrix(fit)
