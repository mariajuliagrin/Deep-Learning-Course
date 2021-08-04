import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score
previsores = pd.read_csv('./files/entradas_breast.csv')#Atributos previsores
classe = pd.read_csv('./files/saidas_breast.csv')#as respostas(classes)

#Dividir em treinamento e teste
#75%treinamento e 25% teste
previsores_treinamento, previsores_teste,classe_treinamento, classe_teste = train_test_split(previsores,classe,test_size = 0.25)

# criar a rede neural
classificador = keras.Sequential()
#UNITS = quantos neurônios na camada oculta
#formula =  (30+1)/2 -> (Número de entradas + numeros de neuronios na saida)/2
# 1 camada oculta - somente
classificador.add(Dense(units=16,activation='relu' , kernel_initializer='random_uniform',input_dim =  30))

#camada de saida
classificador.add(Dense(units=1,activation='sigmoid'))

classificador.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['binary_accuracy',])

#Treinamento
classificador.fit(previsores_treinamento,classe_treinamento,batch_size=10,epochs = 100)

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes>0.5)

precisao =accuracy_score(classe_teste,previsoes)
matriz = confusion_matrix(classe_teste,previsoes)

resultado = classificador.evaluate(previsores_teste,classe_teste)