import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def valid_vac(dados):
    X = dados['Dias'].values.reshape(-1, 1)
    y = dados['Vacina'].values #.reshape(-1, 1)          
    
    # Separando os dados em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)
    
    regressor = SVR(kernel = 'rbf', C=10_000, gamma='scale', epsilon=1, coef0=0)
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    for i in range(len(y_pred)):   
        if y_pred[i] > 100:
            y_pred[i] = 100
        elif y_pred[i] < 0:
            y_pred[i] = 0
    
    # Aplicando as métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    # Retornando as métricas
    print('MÉTRICAS:')
    print(f'- MAE: {mae:.4f}')
    print(f'- RMSE: {rmse:.4f}')
    print(f'- MAPE: {mape:.4f}')
    
    
def valid_uti(dados):
    X = dados['Dias'].values.reshape(-1, 1)
    y = dados['Taxa de Ocupação'].values #.reshape(-1, 1)        
    
    # Separando os dados em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)
    
    regressor = SVR(kernel = 'rbf', C=10_000, gamma='scale', epsilon=1, coef0=0)
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    for i in range(len(y_pred)):   
        if y_pred[i] > 100:
            y_pred[i] = 100
        elif y_pred[i] < 0:
            y_pred[i] = 0
    
    # Aplicando as métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    # Retornando as métricas
    print('MÉTRICAS:')
    print(f'- MAE: {mae:.4f}')
    print(f'- RMSE: {rmse:.4f}')
    print(f'- MAPE: {mape:.4f}')

    

def predict_vac(dados, title):
    # Setando parâmetros para o modelo
    regressor = SVR(kernel = 'rbf', C=10_000, gamma='scale', epsilon=1, coef0=0)
        
    # Selecionado variáveis
    X = dados['Dias'].values.reshape(-1, 1)
    y = dados['Vacina'].values #.reshape(-1, 1)   
    z = dados.iloc[:, 0].values.reshape(-1, 1)

    # Último dia da lista
    last_day = X[len(X)-1][0]
    last_date = z[len(z)-1][0]
    #print(f'Último dia: \033[1;34m{last_day}\033[m, data \033[1;34m{last_date}\033[m')
        
    # Determiando dias para predizer valores futuros
    x1 = np.array([[last_day], [last_day+7], [last_day+14], [last_day+21], [last_day+28]])
     
    # Treinando o modelo
    regressor.fit(X, y)

    # Predizendo para valores de X futuros
    # ------- VALOR QUE SERÁ RETORNADO NA FUNÇÃO -------
    y_pred = regressor.predict(x1)

    XX = np.concatenate((X, x1), axis=0)

    # Create 100 evenly spaced points from smallest X to largest X
    x_range = np.linspace(X.min(), X.max(), 100)

    # Predict y values for our set of X values
    y_svr = regressor.predict(x_range.reshape(-1, 1))

    # Gerando limites para a predição
    for i in range(len(y_svr)):   
        if y_svr[i] > 100:
            y_svr[i] = 100
        elif y_svr[i] < 0:
            y_svr[i] = 0

    # Create 100 evenly spaced points from smallest X to largest X
    xx_range = np.linspace(XX.min(), XX.max(), 100)

    # Predict y values for our set of X values
    yyy_svr = regressor.predict(xx_range.reshape(-1, 1))

    # Gerando limites para a predição
    for i in range(len(y_svr)):   
        if yyy_svr[i] > 100:
            yyy_svr[i] = 100
        elif yyy_svr[i] < 0:
            yyy_svr[i] = 0

    # Create a scatter plot
    fig = px.scatter(dados, x=dados['Dias'], y=dados['Vacina'], 
                     opacity=0.8, color_discrete_sequence=['black'])

    # Add a best-fit line
    fig.add_traces(go.Scatter(x=xx_range, y=yyy_svr, name='SVR Prediction', line=dict(color='blue')))
    fig.add_traces(go.Scatter(x=x_range, y=y_svr, name='SVR', line=dict(color='red')))
    fig.add_traces(go.Scatter(x=x_range, y=y_svr+5, name='+epsilon', line=dict(color='red', dash='dot')))
    fig.add_traces(go.Scatter(x=x_range, y=y_svr-5, name='-epsilon', line=dict(color='red', dash='dot')))
    #fig.add_traces(go.Scatter(x=0, y=0, name=f'{y_pred[1]}', line=dict(color='white')))

    # Change chart background color
    fig.update_layout(dict(plot_bgcolor = 'white'))

    # Update axes lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                     showline=True, linewidth=1, linecolor='black')

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                     showline=True, linewidth=1, linecolor='black')

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                     showline=True, linewidth=1, linecolor='black')

    # Set figure title
    fig.update_layout(title=dict(text=f"   {title}", 
                                 font=dict(color='black')))

    # Update marker size
    fig.update_traces(marker=dict(size=6))

    fig.show()

    return y_pred


def predict_uti(dados, title):
    # Setando parâmetros para o modelo
    regressor = SVR(kernel = 'rbf', C=10_000, gamma='scale', epsilon=1, coef0=0)
        
    # Selecionado variáveis
    X = dados['Dias'].values.reshape(-1, 1)
    y = dados['Taxa de Ocupação'].values #.reshape(-1, 1)   
    z = dados.iloc[:, 0].values.reshape(-1, 1)

    # Último dia da lista
    last_day = X[len(X)-1][0]
    last_date = z[len(z)-1][0]
    #print(f'Último dia: \033[1;34m{last_day}\033[m, data \033[1;34m{last_date}\033[m')
        
    # Determiando dias para predizer valores futuros
    x1 = np.array([[last_day], [last_day+7], [last_day+14], [last_day+21], [last_day+28]])
     
    # Treinando o modelo
    regressor.fit(X, y)

    # Predizendo para valores de X futuros
    # ------- VALOR QUE SERÁ RETORNADO NA FUNÇÃO -------
    y_pred = regressor.predict(x1)

    XX = np.concatenate((X, x1), axis=0)

    # Create 100 evenly spaced points from smallest X to largest X
    x_range = np.linspace(X.min(), X.max(), 100)

    # Predict y values for our set of X values
    y_svr = regressor.predict(x_range.reshape(-1, 1))

    # Gerando limites para a predição
    for i in range(len(y_svr)):   
        if y_svr[i] > 100:
            y_svr[i] = 100
        elif y_svr[i] < 0:
            y_svr[i] = 0

    # Create 100 evenly spaced points from smallest X to largest X
    xx_range = np.linspace(XX.min(), XX.max(), 100)

    # Predict y values for our set of X values
    yyy_svr = regressor.predict(xx_range.reshape(-1, 1))

    # Gerando limites para a predição
    for i in range(len(y_svr)):   
        if yyy_svr[i] > 100:
            yyy_svr[i] = 100
        elif yyy_svr[i] < 0:
            yyy_svr[i] = 0

    # Create a scatter plot
    fig = px.scatter(dados, x=dados['Dias'], y=dados['Taxa de Ocupação'], 
                     opacity=0.8, color_discrete_sequence=['black'])

    # Add a best-fit line
    fig.add_traces(go.Scatter(x=xx_range, y=yyy_svr, name='SVR Prediction', line=dict(color='blue')))
    fig.add_traces(go.Scatter(x=x_range, y=y_svr, name='SVR', line=dict(color='red')))
    fig.add_traces(go.Scatter(x=x_range, y=y_svr+5, name='+epsilon', line=dict(color='red', dash='dot')))
    fig.add_traces(go.Scatter(x=x_range, y=y_svr-5, name='-epsilon', line=dict(color='red', dash='dot')))
    #fig.add_traces(go.Scatter(x=0, y=0, name=f'{y_pred[1]}', line=dict(color='white')))

    # Change chart background color
    fig.update_layout(dict(plot_bgcolor = 'white'))

    # Update axes lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                     showline=True, linewidth=1, linecolor='black')

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                     showline=True, linewidth=1, linecolor='black')

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                     showline=True, linewidth=1, linecolor='black')

    # Set figure title
    fig.update_layout(title=dict(text=f"   {title}", 
                                 font=dict(color='black')))

    # Update marker size
    fig.update_traces(marker=dict(size=6))

    fig.show()

    return y_pred
