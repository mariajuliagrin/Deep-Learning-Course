import numpy as np

#função degrau- para problemas linearmente separaveis
def stepFunction(soma):
    if(soma>=1):
        return 1
    return 0

#função sigmoide -  retorna valores entre 0 e 1
#problemas de classificação binaria
def sigmoidFunction(soma):
    return 1/( 1 + np.exp(-soma))

#função tangente hiperbólica - retorna valores entre -1 e 1

def tahnFuntion(soma):
    return (np.exp(soma) - np.exp(-soma))/(np.exp(soma)+ np.exp(-soma))

#função Relu - retorna valores maiores ou igual a zero Y=max(0,x)
#muito utilizada em redes convolucionais e redes com muitas camadas
def relu(soma):
    if(soma>=0):
        return soma
    return 0

#função linear
def linearFuntion (soma):
    return soma

#função softmax - retorna probabilidades
#problemas de classificação com mais de 2 classes
def softmax(x):
    ex = np.exp(x)
    return ex / ex.sum()
    

teste = stepFunction(-1)
teste = sigmoidFunction(0.358)
teste = tahnFuntion(-0.358)
teste = relu(100)
teste = linearFuntion(20)

valores = [5,2,1.3]#retornados da camada de saida
print(softmax(valores))#probabilidades para cada um deles