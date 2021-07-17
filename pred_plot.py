# ----------------- PARTE 1 - PREDIÇÃO DA VACINAÇÃO  ---------------------------
# Os dados gerados por esta predição serão usados
# na predição de ocupação de leitos de UTI

from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Setando parâmetros para o modelo
regressor = SVR(kernel = 'poly', C=1, gamma='auto', degree=8, epsilon=.101, coef0=1)
    
# Selecionado variáveis
X = vacina.iloc[:, 1].values.reshape(-1, 1)
y = vacina.iloc[:, 2].values.reshape(-1, 1)   
z = vacina.iloc[:, 0].values.reshape(-1, 1)

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
X_pred = X[len(vacina):]
    
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

# ----------------- RESULTADO DA PREDIÇÃO ------------------------ 
y_pred






# --------- PARTE 2 - PREDIÇÃO DA OCUPAÇÃO DE LEITOS DE UTI  ---------------------------
# Usando a predição da taxa de vacinação para
# predizer a taxa de ocupação de leitos de UTI

from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Setando parâmetros para o modelo
regressor = SVR(kernel = 'poly', C=1, gamma='auto', degree=8, epsilon=.101, coef0=1)
    
# Selecionado variáveis
X1 = brasil.iloc[:, 1].values#.reshape(-1, 1)
X2 = brasil.iloc[:, 3].values#.reshape(-1, 1)
y = brasil.iloc[:, 2].values.reshape(-1, 1)   
z = brasil.iloc[:, 0].values.reshape(-1, 1)


# Último dia da lista
last_day = X1[len(X1)-1]#[0]
last_date = z[len(z)-1]#[0]
#print(f'Último dia: \033[1;34m{last_day}\033[m, data \033[1;34m{last_date}\033[m') 
# Determiando dias para predizer valores futuros
x1 = np.array([last_day, last_day+7, last_day+14, last_day+21, last_day+28])  
# Inserindo os dias para predizer dados futuros no array
X1 = np.concatenate((X1, x1), axis=0)


# Determiando dias para predizer valores futuros
x2 = np.array([y_pred[0], y_pred[1], y_pred[2], y_pred[3], y_pred[4]]) 
# Inserindo os dias para predizer dados futuros no array
X2 = np.concatenate((X2, x2), axis=0)

index = np.array(range(len(X1)))
W = pd.DataFrame({'Data': X1, 'Vacina': X2, 'index': index})
W = W.set_index(['index'])

# Normalizando os dados
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(W)
y = sc_y.fit_transform(y)

# Atribuindo valores a serem previstos
X_pred = X[len(brasil):]
X_tran = X[:len(brasil)]

# Desacoplando valores futuros de X (dias)
X = np.delete(X, len(X)-1, 0)
X = np.delete(X, len(X)-1, 0)
X = np.delete(X, len(X)-1, 0)
X = np.delete(X, len(X)-1, 0)
X = np.delete(X, len(X)-1, 0)

# Treinando o modelo
regressor.fit(X, y)

# Separando as varáveis independentes 
AX, AY = list(), list()
for i in range(len(X)):
    AX.append(X[i][0]) 
    AY.append(X[i][1])

# ---------------------_ VALORES ATUAIS -----------------
# Predizendo para valores de X futuros
Y_tran = regressor.predict(X_tran)
#Y_tran = sc_y.inverse_transform(Y_tran)

AZ = list(Y_tran)

# ---------------------_ VALORES FUTUROS -----------------
# Predizendo para valores de X futuros
Y_pred = regressor.predict(X_pred)
Y_pred = sc_y.inverse_transform(Y_pred)

# ---------------  RESULTADO DA PREDIÇÃO ------------------ 
Y_pred






# --------------------- PARTE 3 - PLOTANDO GRÁFICO  ---------------------------
# ----------- For creating a prediciton plane to be used in the visualization -----------
# Set Increments between points in a meshgrid
mesh_size = 1

# Identify min and max values for input variables
x_min, x_max = min(AX), max(AX)               #AX.min(), AX.max()
y_min, y_max =  min(AY), max(AY)              #AY.min(), AY.max()

# Return evenly spaced values based on a range between min and max
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)

# Create a meshgrid
xx, yy = np.meshgrid(xrange, yrange)



# ----------- Create a prediciton plane  -----------
# Use models to create a prediciton plane --- SVR
pred_svr = regressor.predict(np.c_[xx.ravel(), yy.ravel()])
pred_svr = pred_svr.reshape(xx.shape)

# Observe, .ravel () nivela a matriz para uma matriz 1D,
# então np.c_ pega elementos de arrays xx e yy achatados e os coloca juntos,
# isso cria a forma certa necessária para a entrada do modelo

# array de predição que é criado pela saída do modelo é um array 1D,
# Portanto, precisamos reformulá-lo para ter a mesma forma de xx ou yy para poder exibi-lo em um gráfico
# ----------- For creating a prediciton plane to be used in the visualization -----------
# Set Increments between points in a meshgrid


# Create a 3D scatter plot with predictions
fig = px.scatter_3d(vacina, x=AX, y=AY, z=AZ, 
                 opacity=0.8, color_discrete_sequence=['black'])

# Set figure title and colors
fig.update_layout(title_text="(%) Ocupação de Leitos - Gráfico de dispersão 3D",
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
fig.update_traces(marker=dict(size=6))

# Add prediction plane
fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred_svr, name='SVR',
                          colorscale=px.colors.sequential.Plotly3))

fig.show()
