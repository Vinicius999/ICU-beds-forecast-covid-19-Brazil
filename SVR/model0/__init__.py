import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


def predict(dados, title):
    # Setando parâmetros para o modelo
    regressor = SVR(kernel = 'poly', C=1, gamma='auto', epsilon=0.101, coef0=1)

    # Selecionado variáveis
    X = dados.iloc[:, 1].values.reshape(-1, 1)
    y = dados.iloc[:, 2].values.reshape(-1, 1)
    z = dados.iloc[:, 0].values.reshape(-1, 1)

    # Último dia da lista
    last_day = X[len(X)-1][0]
    last_date = z[len(z)-1][0]
    #print(f'Último dia: \033[1;34m{last_day}\033[m, data \033[1;34m{last_date}\033[m')
    
    # Determiando dias para predizer valores futuros
    x1 = np.array([[last_day], [last_day+7], [last_day+14], [last_day+21], [last_day+28]])
    
    # Inserindo os dias para predizer dados futuros no array
    X = np.concatenate((X, x1), axis=0)
    
    # Normalizando os dados
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)
    
    # Atribuindo valores a serem previstos
    X_pred = X[len(dados):]
    
    # Desacoplando valores futuros de X (dias)
    X = np.delete(X, len(X)-1, 0)
    X = np.delete(X, len(X)-1, 0)
    X = np.delete(X, len(X)-1, 0)
    X = np.delete(X, len(X)-1, 0)
    X = np.delete(X, len(X)-1, 0)
    
    # Treinando o modelo
    regressor.fit(X, y)
    
    # Predizendo para valores de X futuros
    y_pred = regressor.predict(X_pred)
    y_pred = sc_y.inverse_transform(y_pred)
    
    X_grid = np.arange(min(X), max(X), 0.01) #curva suave
    X_grid = X_grid.reshape((len(X_grid), 1))

    X_pred = np.arange(min(X_pred), max(X_pred), 0.01) #curva suave
    X_pred = X_pred.reshape((len(X_pred), 1))
    
    y_plot = regressor.predict(X_pred)
    
    # Gerando limites para a predição
    for i in range(len(y_plot)):   
        if y_plot[i].any() > 100:
            y_plot[i] = 100
        elif y_plot[i].any() < 0:
            y_plot[i] = 0

    # Visualizando os dados 
    plt.scatter(X, y, color = 'red')
    plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
    plt.plot(X_pred, y_plot, color = 'red')
    plt.title('\nSVR - RBF (kernel = "rbf", C=10000, gamma="scale", epsilon=1, coef0=0)\n\n'
              f'{title}\n\n'
              f'7 dias à frente: {y_pred[1]:.2f}%\n'
              f'14 dias à frente: {y_pred[2]:.2f}%\n'
              f'21 dias à frente: {y_pred[3]:.2f}%\n'
              f'28 dias à frente: {y_pred[4]:.2f}%\n')
    plt.xlabel('data')
    plt.ylabel('taxa de ocupação')
    plt.show()
    