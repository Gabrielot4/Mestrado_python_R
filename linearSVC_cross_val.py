import pandas as pd
import numpy as np
import time
import string

inicio = time.time()                    # começa a contar o tempo de processamento

data = pd.read_excel('Consolidado_Geral_Modif3.xlsx', sheet_name='c_valor_s_orgao2', index_col=None)
data = pd.DataFrame(data)
print('\n \033[1;30;43m Visualização do DatFrame: \033[m \n ', data.head())
print('\n \033[1;30;43m Tipos de dados do DatFrame: \033[m \n ',data.dtypes)


### PREPARE THE DATA - REMOVE SPACE, PUNCTUATION, ALFANUMERICS and LOWER
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


print('\n \033[1;30;43m Dados Ausentes: \033[m \n')
print(data.isnull().sum(),'\n')

# comarcas com baixa frequência (< 100) retiradas do dataset
print('\n \033[1;30;43m Frequência Comarca: \033[m \n')
print(data['Comarca'].value_counts())
mask = data['Comarca'].value_counts().head(133).index                                                                               # mask fica com os 133 valores da coluna Comarcas com maior frequência (> 100)
data = data.loc[data['Comarca'].isin(mask)]                                                                                         # datafram fica com todos os 134 valores que mais aparecem
print('\n \033[1;30;43m 134 maiores frequências Comarca: \033[m \n')
print(data['Comarca'].value_counts())
print(data.shape)

# exportar para excel a base pré-processada
#data.to_excel('base_regressao_logistica.xlsx')


## SEPARATE INPUTS AND OUTPUTS
X = data.drop(['Valor_Pago'], axis='columns')                                                                           # X são os dados dos atributos de entrada (variáveis independentes, features)
y = data['Valor_Pago']                                                                                                  # y é a o valor alvo  aser predito (classe, variável dependente)


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

#Parâmetros a serem testados
c = np.array([0.1,0.5,1,2,3])
penalty = ['l2']
loss = ['hinge', 'squared_hinge']
params_grid = {'C':c, 'loss':loss, 'penalty':penalty}

## MODEL
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier

svm_linearSVC = LinearSVC(max_iter=100000,random_state=42)  #ovo = one vs one: faz uma combinação das classes

# Validação cruzada
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

gridSVC = GridSearchCV(svm_linearSVC, param_grid=params_grid, cv=5, n_jobs=-1)
gridSVC.fit(X,y)

print('C: ', gridSVC.best_estimator_.C)
print('Loss: ', gridSVC.best_estimator_.loss)
print('Penalty: ', gridSVC.best_estimator_.penalty)
print('Acurácia: ', gridSVC.best_score_)


fim = time.time()                       # fim do tempo de processamento
print('\n')
print('\033[1;30;42m Tempo de processamento [min] =  \033[m', (fim-inicio)/60)