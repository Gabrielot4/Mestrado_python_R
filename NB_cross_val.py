import pandas as pd
import numpy as np
import time
import string
#import unidecode

inicio = time.time()                    # começa a contar o tempo de processamento

data = pd.read_excel('Consolidado_Geral_Modif3.xlsx', sheet_name='c_valor_s_orgao2', index_col=None)
data = pd.DataFrame(data)
print('\n \033[1;30;43m Visualização do DatFrame: \033[m \n ', data.head())
print('\n \033[1;30;43m Tipos de dados do DatFrame: \033[m \n ',data.dtypes)



### PREPARE THE DATA - REMOVE SPACE, PUNCTUATION, ALFANUMERICS and LOWER, ACCENTS
import re

punctuation = lambda x: re.sub('[%s]' % re.escape(string.punctuation), "", x)                                                       # remove pontuações de qualquer tipo das strings e as substitui por nada ""
alphanumeric = lambda x: re.sub(r"""\w*\d\w*""", "", x)                                                                             # substitui textos alfanumericos das strings por nada ""

data = [data[col].map(punctuation).map(alphanumeric).str.replace(" ","_").str.lower() for col in data.columns]                      # aplica as funções lambda acima em cada coluna, conforma o loop itera sobre as colunas do dataframe, e deixa tudo minúsculo
data = pd.DataFrame(data).transpose()                                                                                               # precisa transpor, senão os atributos viram linhas e vice-versa
cols = data.columns
data[cols] = data[cols].apply(lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))           # remove os acentos as plavras
print('\n \033[1;30;43m DatFrame pré-processado: sem pontuação, em minúsculo, sem espaços, sem números,: \033[m \n ',data.head())

# embaralhar dataset
data = data.sample(frac=1)
print('\n \033[1;30;43m DatFrame embaralhado: \033[m \n ',data.head())

# dados null
print('\n \033[1;30;43m Dados Ausentes: \033[m \n')
print(data.isnull().sum(),'\n')

# comarcas com baixa frequência (< 100) retiradas do dataset
print('\n \033[1;30;43m Frequência Comarca: \033[m \n')
print(data['Comarca'].value_counts())
mask = data['Comarca'].value_counts().head(133).index                                                                               # mask fica com os 134 valores da coluna Comarcas com maior frequência (> 100)
data = data.loc[data['Comarca'].isin(mask)]                                                                                         # datafram fica com todos os 134 valores que mais aparecem
print('\n \033[1;30;43m 134 maiores frequências Comarca: \033[m \n')
print(data['Comarca'].value_counts())
print(data.shape)


## SEPARATE INPUTS AND OUTPUTS
X = data.drop('Valor_Pago', axis='columns')
y = data['Valor_Pago']


## PRÉ-PROCESSAMENTO DOS DADOS - TRANSFORMAR DADOS CATEGÓRICOS PARA NUMÉRICOS
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Transformando os atritbutos por OneHotEnconder
ohe = OneHotEncoder(handle_unknown='ignore')                                                                            # esse argumento é colocado para ignorar possíveis categorias que estão no X_train e não estão no X_test. Se no X_test aparece alguma coisa que não tem no X_train, dá erro.
ohe.fit(X)                                                                                                        # transformando os atributos categóricos em atributos numéricos: Ex: Niteroi = 1, Brasilia = 2, ...
X = ohe.transform(X).toarray()
print('\n \033[1;30;44m Atributos encoded: \033[m \n ', X)

# Transformando a classe a ser predita por LabelEncoder
le = LabelEncoder()
le.fit(y)
y = le.transform(y)
print('\n \033[1;30;44m Class encoded: \033[m \n ', y)

#Parâmertro alpha
alphas = np.array([0.1, 0.01, 0.001, 0.0001])
params = {'alpha': alphas}

## MODEL
from sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB

cnb = CategoricalNB()                       # aplciar modelo de Naive Bayes para dados categoricos
gnb = GaussianNB()
mnb = MultinomialNB()

# Validação cruzada
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

gridMNB = GridSearchCV(cnb, param_grid={'alpha':[1,0.01,0.005, 0.001, 0.0001]}, cv=5, n_jobs=-1)                  # Quanto maior o alpha, mais se aproxima de uma distribuição uniforma
gridMNB.fit(X, y)

print('Alpha: ', gridMNB.best_estimator_.alpha)
print('Acurácia: ', gridMNB.best_score_)



fim = time.time()                       # fim do tempo de processamento
print('\n')
print('\033[1;30;42m Tempo de processamento =  \033[m', fim-inicio)
