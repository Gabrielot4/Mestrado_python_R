### MODELO DE REGRESS�O LOG�STICA MULTINOMIAL PARA A DISSERTA��O DE MESTRADO

### Resumo dos testes
# anova function: compares two nested models using the likelihood ratio test, test if the betas coefs are equal zero, except the intercept
# chisq function: compares if there's significant difference between the expected and observed frequencies
# lrtest function: tests if the variable has significant effects on the model (if just one variable is drop from the full model)
#                  tests if the full model is better than the intercept only model, it means, the full model explains the outcomes better


dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")

setwd("D:/15.1 Mestrado/1.0 - Disserta��o de mestrado/2. python_mestrado")     # diret�rio onde est� arquivo

#library(broom)
#library(openxlsx)
library(readxl)
data <- read_excel("base_reduzido.xlsx")
View(data)
summary(data)


#Excluir colunas que n�o interessam
data$Cia_dummy <- NULL
data$M�s <- NULL
data$M�s_ <- NULL
data$Ano <- NULL
data$Comarca <- NULL
data$UF <- NULL
data$Orgao <- NULL
data$`Dano Moral` <- NULL
data$`Dano Material` <- NULL
data$`Valor Pago` <- NULL

View(data)



#Tranformando vari�veis categ�ricas em fator e definindo as refer�ncias com base nas maiores frequ�ncias
data$Cia <- relevel(as.factor(data$Cia), ref = 'Latam') 
data$M�s__ <- relevel(as.factor(data$M�s__), ref = 'nf') # ref = nf (meses de nao ferias)
data$Ano_ <- relevel(as.factor(data$Ano_), ref='ano_cinco') # ref = 2020
data$Regi�o <- relevel(as.factor(data$Regi�o), ref='SE')
data$Dano_Moral_red <- relevel(as.factor(data$Dano_Moral_red), ref='medio') 
data$Dano_Material_red <- relevel(as.factor(data$Dano_Material_red), ref='baixo') 
data$motivo <- relevel(as.factor(data$motivo), ref='cancelamento_voo')
data$causa <- relevel(as.factor(data$causa), ref='culpa_cia')
data$andamento <- relevel(as.factor(data$andamento), ref='procedente')
data$Valor_pago_red <- relevel(as.factor(data$Valor_pago_red), ref='medio')


### Modeling
### Multinomial Logistic Regression

#install.packages("nnet")
library(nnet)

# mymodel: modelo com todos os regressores
mymodel <- multinom(Valor_pago_red ~., data = data)               # valor_pago_red ~. : valor_pago_red � a dependente e ~. significa que todas as outras colunass s�o as independtens
coefs <- summary(mymodel)      # mostra os coeficientes para os dados de treino. Esses coeficientes s�o os logaritmos das raz�es de chances
coefs


odds_ratio <- exp(summary(mymodel)$coefficients)  # raz�o de chances dos coeficientes estimandos: exp(ln(Betas))
odds_ratio

# mymodel_oi: modelo s� com intercepto
mymodel_oi <- multinom(Valor_pago_red ~1, data=data)
summary(mymodel_oi)

# modelo sem uma vari�vel espec�fica: s� para testar o lrtest entre modelo completo e esse modelo sem a vari�vel e ver se os resultados batem com o loop for l� no final do c�digo 
modelo_sem_ano <- multinom(Valor_pago_red ~Cia + M�s__ + Regi�o + Dano_Material_red + Dano_Moral_red + motivo + causa + andamento, data=data)
summary(modelo_sem_ano)

### 2-tailed z-test
z <- summary(mymodel)$coefficients/summary(mymodel)$standard.errors
z
p <- (1 - pnorm(abs(z), 0, 1))*2  
p

### Predicted probabilities 
head(pp.values <- mymodel$fitted.values) # Probabilidade de prever baix ou medio ou alto, em n�mero, para cada linha do dataframe
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

### Likelihood-ratio test (chisq statistic): para rejeitar ou n�o a hip�tese de os coefs dos regressores serem iguais a zer
### H0: B1 = B2 = Bj = 0 | se p-value < 0.01, ent�o rejeita-se HO
### Model fit information: anova function compares two nested models using likelihood ratio test
anova(mymodel_oi, mymodel) # rejeita-se H0, ou seja, os regressores do modelo completo t�m influ�ncia nas respostas da vari�vel dependente


### Goodness of fit: dizer no texto do mestrado que o teste chi quadrado aqui � para goodness of fit, posi existe variso outros para outras coisas
### Testar se existe diferen�a estatisticamente significante entre os valores esperados e os observados
### H0: n�o existe diferen�a significante entre os valores observados e esperados
### H1: existe diferen�a
### H0: (null hypothesis) A variable follows a hypothesized distribution.
### H1: (alternative hypothesis) A variable does not follow a hypothesized distribution.
### p-value < alpha: rejeita HO, ou seja, n�o? h� evid�ncias significantes de que existe diferen�as entres os valores observados e esperados
observed <- table(pp.outcome)     # frequ�ncias observados, ou seja, s�o as predi��es do modelo de regress�o log�sticas
expected <- table(data$Valor_pago_red)    # frequ�ncias esperadas, que est�o na base de dados
chi2 <- chisq.test(observed, p = c(medio=.343789, alto=.319777, baixo=.336434)) # chisq.test(observed freq, expected freq.)
chi2 # p-value < 0,01: rejeita H0, ou seja, os dados preditos diferem dos dados esperados. Isso pode dizer que a acur�cia do modelo n�o foi t�o boa.
chi2$observed
chi2$expected
chi2$residuals
help("chisq.test")

### Outra forma do Goodness of fit, mais f�cil de compreender o resultado. H0: os valores observados possuem frequ�ncias parecidas com as especificadas
install.packages("lsr")
library(lsr)

help("goodnessOfFitTest")

goodnessOfFitTest(pp.outcome, p = c(baixo=.3364, medio=.3438, alto=.3198)) # goodnessOfFitTest(observado, esperado) | os valores esperados s�o as propor��es das categorias presentes na base de dados original

### Chi-square independence test, utiliza o mesmo chisq.text acima, mas as entras s�o as duas vari�veis que ser analisar.
### Testar se h� associa��o entre duas vairi�veis categ�ricas, ou seja, se s�o independentes ou n�o
# H0: n�o h� associa��o entre as vari�veis preditoras e a dependente
# H1: h� associa��o entre as vari�veis preditoras e a dependente
# p < 5%: rejeita H0, ou seja, h� associa��o entre as vari�veis


### Log Likelihood ratio test
### Analisar quais preditores permitem prever a categoria de resposta, ou seja, se os regressores t�m efeitos significantes no modelo
### HO: coeficientes Betas das vari�veis x s�o simultanemaente  zero, exceto o intercepto
library(lmtest)
lrtest(mymodel, mymodel_oi) # compara��o entre  modelo completo e s� com intercepto. p<alpha rejeita H0,ou seja, o modelo completo tem mais acur�cia sob 1% de n�vel de significancia.
lrtest(mymodel, modelo_sem_ano)

help("lrtest")

colunas = c('Cia', 'M�s__', 'Ano_', 'Regi�o', "Dano_Moral_red", "Dano_Material_red", "motivo", "causa", "andamento" )

# signific�ncia de cada preditor no modelo. Pode ser usado para identificar quais vari�veis ajudam na predi��o do modelo
for (i in colunas) {
  lr <- lrtest(mymodel, i)  
  print(lr)
}

### Multicolinearidade
library(car)
vif(mymodel)

### Pseudo  R2
library(DescTools)
PseudoR2(mymodel, which = c("CoxSnell", "Nagelkerke", "McFadden"))



### CrossValidation with caret
library(caret)
fit.control <- trainControl(method = 'repeatedcv', number = 5, repeats = 10)
fit <- train(Valor_pago_red ~., data=data, method = "multinom", trControl = fit.control, trace=FALSE)
fit
confusionMatrix(fit)
