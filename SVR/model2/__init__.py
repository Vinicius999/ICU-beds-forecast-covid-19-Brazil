import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def valid_UTI(dados):
    X = dados[['Dias', 'Vacina']]      
    y = dados['Taxa de Ocupação'].values          
    
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
        
    # Determiando dias para predizer valores futuros
    x1 = np.array([[last_day], [last_day+7], [last_day+14], [last_day+21], [last_day+28]])
     
    # Treinando o modelo
    regressor.fit(X, y)

    # Predizendo para valores de X futuros
    y_pred = regressor.predict(x1)
    

    # -------  GERANDO VALOR DA PREDIÇÃO DA OCUPAÇÃO DE LEITOS --------
    X = dados[['Dias', 'Vacina']]     
    y = dados['Taxa de Ocupação'].values       
    z = dados.iloc[:, 0].values.reshape(-1, 1)
    
    # Último dia da lista
    last_day = X['Dias'][-1]#[0]
    last_date = z[len(z)-1]#[0]
        
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

    XX = np.concatenate((X, X_pred), axis=0)

    # Create 100 evenly spaced points from smallest X to largest X
    x_range = np.linspace(X.min(), X.max(), 100)

    # Predict y values for our set of X values
    Y_svr = regressor.predict(x_range.reshape(-2, 2))

    # Gerando limites para a predição
    '''for i in range(len(Y_svr)):   
        if Y_svr[i] > 100:
            Y_svr[i] = 100
        elif Y_svr[i] < 0:
            Y_svr[i] = 0'''

    # Create 100 evenly spaced points from smallest X to largest X
    xx_range = np.linspace(XX.min(), XX.max(), 100)

    # Predict y values for our set of X values
    YY_svr = regressor.predict(xx_range.reshape(-2, 2))

    # Gerando limites para a predição
    '''for i in range(len(Y_svr)):   
        if YY_svr[i] > 100:
            YY_svr[i] = 100
        elif YY_svr[i] < 0:
            YY_svr[i] = 0'''

    # Create a scatter plot
    fig = px.scatter(dados, x=dados['Dias'], y=dados['Taxa de Ocupação'], 
                     opacity=0.8, color_discrete_sequence=['black'])

    # Add a best-fit line
    fig.add_traces(go.Scatter(x=xx_range, y=YY_svr, name='SVR Prediction', line=dict(color='blue')))
    fig.add_traces(go.Scatter(x=x_range, y=Y_svr, name='SVR', line=dict(color='red')))
    fig.add_traces(go.Scatter(x=x_range, y=Y_svr+5, name='+epsilon', line=dict(color='red', dash='dot')))
    fig.add_traces(go.Scatter(x=x_range, y=Y_svr-5, name='-epsilon', line=dict(color='red', dash='dot')))
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

    return Y_pred
    


def plot_UTI_3D(dados, title):
    # -------  GERANDO VALOR DA PREDIÇÃO DA VACINAÇÃO --------
    # Setando parâmetros para o modelo
    regressor = SVR(kernel = 'rbf', C=10_000, gamma='scale', epsilon=1, coef0=0)

    X = dados[['Dias', 'Vacina']]      
    y = dados['Taxa de Ocupação'].values      
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
    fig = px.scatter_3d(dados, x=dados['Dias'], y=dados['Vacina'], z=dados['Taxa de Ocupação'], 
                     opacity=0.8, color_discrete_sequence=['black'])
            
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


    
def predict_UTI_3D(dados, title):
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
        
    # Determiando dias para predizer valores futuros
    x1 = np.array([[last_day], [last_day+7], [last_day+14], [last_day+21], [last_day+28]])
     
    # Treinando o modelo
    regressor.fit(X, y)

    # Predizendo para valores de X futuros
    y_pred = regressor.predict(x1)
    

    # -------  GERANDO VALOR DA PREDIÇÃO DA OCUPAÇÃO DE LEITOS --------
    X = dados[['Dias', 'Vacina']]     
    y = dados['Taxa de Ocupação'].values       
    z = dados.iloc[:, 0].values.reshape(-1, 1)
    
    # Último dia da lista
    last_day = X['Dias'][-1]#[0]
    last_date = z[len(z)-1]#[0]
        
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
    fig = px.scatter_3d(dados, x=dados['Dias'], y=dados['Vacina'], z=dados['Taxa de Ocupação'], 
                     opacity=0.8, color_discrete_sequence=['black'])
            
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
