import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVR


def plot_UTI(dados, title):
    # -------  GERANDO VALOR DA PREDIÇÃO DA VACINAÇÃO --------
    # Setando parâmetros para o modelo
    regressor = SVR(kernel = 'rbf', C=10_000, gamma='scale', epsilon=1, coef0=0)

    X = dados[['Dias', 'Vacina']]      #.values.reshape(-1, 1)
    y = dados['TaxaOcup'].values       #.reshape(-1, 1)   
    z = dados.iloc[:, 0].values.reshape(-1, 1)

    # Treinando o modelo
    regressor.fit(X, y)


    # ----------- GERANDO GRÁFICO -----------------------------------
    # ----------- For creating a prediciton plane to be used in the visualization -----------
    # Set Increments between points in a meshgrid
    mesh_size = 1

    # Identify min and max values for input variables
    x_min, x_max = X['Dias'].min(), X['Dias'].max()
    y_min, y_max = X['Vacina'].min(), X['Vacina'].max()

    # Return evenly spaced values based on a range between min and max
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)

    # Create a meshgrid
    xx, yy = np.meshgrid(xrange, yrange)

    # ----------- Create a prediciton plane  -----------
    # Use models to create a prediciton plane --- SVR
    pred_svr = regressor.predict(np.c_[xx.ravel(), yy.ravel()])
    #pred_svr = pred_svr.reshape(xx.shape)

    # Gerando limites para a predição
    for i in range(len(pred_svr)):   
        if pred_svr[i] > 100:
            pred_svr[i] = 100
        elif pred_svr[i] < 0:
            pred_svr[i] = 0

    pred_svr = pred_svr.reshape(xx.shape)

    # Create a 3D scatter plot with predictions
    fig = px.scatter_3d(dados, x=dados['Dias'], y=dados['Vacina'], z=dados['TaxaOcup'], 
                     opacity=0.8, color_discrete_sequence=['black'])
                     # brasil, x=brasil['Dias'], y=brasil['Vacina'], z=brasil['TaxaOcup']
                     # brasil, x=AX, y=AY, z=AZ
            
    # Set figure title and colors
    fig.update_layout(title_text=f"(%) Ocupação de Leitos de UTI com base na Vacinação - {title}",
                      scene = dict(xaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='lightgrey'),
                                   yaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='lightgrey'),
                                   zaxis=dict(backgroundcolor='white',
                                              color='black', 
                                              gridcolor='lightgrey')))
    # Update marker size
    fig.update_traces(marker=dict(size=3))

    # Add prediction plane
    fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred_svr, name='SVR',
                              colorscale=px.colors.sequential.Plotly3))

    fig.show()


    
def predict_UTI(dados, title):
    # -------  GERANDO VALOR DA PREDIÇÃO DA VACINAÇÃO --------
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
    y_pred = regressor.predict(x1)
    

    # -------  GERANDO VALOR DA PREDIÇÃO DA OCUPAÇÃO DE LEITOS --------
    X = dados[['Dias', 'Vacina']]      #.values.reshape(-1, 1)
    y = dados['TaxaOcup'].values       #.reshape(-1, 1)   
    z = dados.iloc[:, 0].values.reshape(-1, 1)
    
    # Último dia da lista
    last_day = X['Dias'][-1]#[0]
    last_date = z[len(z)-1]#[0]
    #print(f'Último dia: \033[1;34m{last_day}\033[m, data \033[1;34m{last_date}\033[m')
        
    # Determiando dias para predizer valores futuros
    x1 = np.array([last_day, last_day+7, last_day+14, last_day+21, last_day+28])
    # Valores futuros de vacinação
    x2 = np.array([y_pred[0], y_pred[1], y_pred[2], y_pred[3], y_pred[4]])

    X_plot = pd.DataFrame({'Dias': X['Dias'], 'Vacina': X['Vacina']})
    X_pred = pd.DataFrame({'Dias': x1, 'Vacina': x2})
    
    cont = 1
    for i in range(len(X_plot), len(X_plot)+4):
        X_plot.loc[i, 'Dias']  = X_pred['Dias'][cont]
        X_plot.loc[i, 'Vacina'] = X_pred['Vacina'][cont]
        cont += 1 

    # Treinando o modelo
    regressor.fit(X, y)

    # Predizendo para valores de X futuros
    # ----- VALOR QUE A FUNÇÃO IRÁ RETORNAR -----
    Y_pred = regressor.predict(X_pred)

    
    # ----------- PLOTAGEM DA PREDIÇÃO -------------------
    # ----------- For creating a prediciton plane to be used in the visualization -----------
    # Set Increments between points in a meshgrid
    mesh_size = 1

    # Identify min and max values for input variables
    x_min, x_max = X_plot['Dias'].min(), X_plot['Dias'].max()
    y_min, y_max = X_plot['Vacina'].min(), X_plot['Vacina'].max()

    # Return evenly spaced values based on a range between min and max
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)

    # Create a meshgrid
    xx, yy = np.meshgrid(xrange, yrange)

    # ----------- Create a prediciton plane  -----------
    # Use models to create a prediciton plane --- SVR
    pred_svr = regressor.predict(np.c_[xx.ravel(), yy.ravel()])

    # Gerando limites para a predição
    for i in range(len(pred_svr)):   
        if pred_svr[i] > 100:
            pred_svr[i] = 100
        elif pred_svr[i] < 0:
            pred_svr[i] = 0

    pred_svr = pred_svr.reshape(xx.shape)

    # Create a 3D scatter plot with predictions
    fig = px.scatter_3d(dados, x=dados['Dias'], y=dados['Vacina'], z=dados['TaxaOcup'], 
                     opacity=0.8, color_discrete_sequence=['black'])
                     # brasil, x=brasil['Dias'], y=brasil['Vacina'], z=brasil['TaxaOcup']
                     # brasil, x=AX, y=AY, z=AZ
            
    # Set figure title and colors
    fig.update_layout(title_text=f"(%) Ocupação de Leitos de UTI com base na Vacinação - {title}",
                      scene = dict(xaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='lightgrey'),
                                   yaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='lightgrey'),
                                   zaxis=dict(backgroundcolor='white',
                                              color='black', 
                                              gridcolor='lightgrey')))
    # Update marker size
    fig.update_traces(marker=dict(size=3))

    # Add prediction plane
    fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred_svr, name='SVR'))#,
                              #colorscale=px.colors.sequential.Plotly3))

    fig.show()

    return Y_pred