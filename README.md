# Predicting ICU bed occupancy rate by COVID-19 in Brazil and Regions using SVR based on vaccination.

This study sought to apply the SVR technique to predict the ICU bed occupancy rate by COVID-19 in Brazil for 7, 14, 21 and 28 days after July 19, 2021. See the document [here](https://jhi.sbis.org.br/index.php/jhi-sbis/article/view/919/508).

---

## Data

- Data Source:  https://bigdata-covid19.icict.fiocruz.br/

- Number of instances:  37 instances

- Date of last instance:  19/07/2021 (day / month / year)

- Final data:

  ![Figure](https://github.com/Vinicius999/ICU-beds-forecast-covid-19-Brazil/blob/main/images/dataset-image.png)

---

## Methodology

- Training data:  85%
- Validation data:  15%
- Metrics:  MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error)
- No. of training and validation tests:  5 tests using Grid Search
- Prediction time intervals:  7, 14, 21 and 28 days after the last collection date

---

## Results

*en*: All results shown in the figures below start from January 17, 2020 (day 0) e goes up to 28 days after the last date available on the dataset, which is July 19, 2021.

*pt-br*: Os resultados mostrados nas figuras abaixo iniciam em 17 de janeiro de 2020 (dia 0) e vão até a última data disponível no banco de dados, que é 19 de julho de 2021.

#### Figure 1

![image1](https://user-images.githubusercontent.com/87482328/143949758-2e558a6b-21c7-42ca-8542-eafeab904378.png)

*en*: Figure 1 shows the vaccination rate x dates for Brazil. The real data is shown as red circles, while the blue curve presents the result of the regression and the red curve presents the predicted vaccination rate for 7, 14, 21 and 28 days ahead.

*pt-br*: A Figura 1 apresenta o gráfico da taxa de vacinação em relação a data para os dados do Brasil. Os dados reais são mostrados como círculos vermelhos, a curva azul representa o resultado da regressão e a curva em vermelho representa a predição da taxa de vacinação para 7, 14, 21 e 28 dias no futuro. 

#### Table 1

| Dias | Taxa de vacinação (%) | Métricas |  Erro  |
| :--: | :-------------------: | :------: | :----: |
|  7   |         21.73         |   MAE    | 0.8139 |
|  14  |         23.04         |   RMSE   | 1.0341 |
|  21  |         24.37         |          |        |
|  28  |         25.71         |          |        |

*en*: Table 1 shows the forecasting results for vaccination rate on Brazil.

*pt-br*: A tabela 1 mostra os resultados da predição da taxa de vacinação no Brasil.

#### Figure 2

![image5](https://user-images.githubusercontent.com/87482328/143949761-82d2a933-9ff6-45e2-b38f-d8b4e3922053.png)

*en*: Figure 2 shows the vaccination rate x dates for the North region of Brazil. The black dots mark the real data from the dataset, the red curve represents the result of the SVR regression, while the blue curve is the prediction result for 7, 14, 21 and 28 days ahead.

*pt-br*: A Figura 2 apresenta o gráfico da taxa de vacinação em relação a data para a região Norte do Brasil. Os pontos pretos são os valores reais do banco de dados, a curva em vermelho representa a regressão da SVR e a curva azul é o resultado da predição para 7, 14, 21 e 28 dias no futuro.

#### Table 2

| Dias | Taxa de vacinação (%) | Métricas |  Erro  |
| :--: | :-------------------: | :------: | :----: |
|  7   |         17.62         |   MAE    | 0.8764 |
|  14  |         17.99         |   RMSE   | 0.9459 |
|  21  |         18.24         |          |        |
|  28  |         18.36         |          |        |

*en*: Table 2 shows the forecasting results for vaccination rate on the North region of Brazil.

*pt-br*: A tabela 2 mostra os resultados da predição da taxa de vacinação na região Norte do Brasil.

#### Figure 3

![image7](https://user-images.githubusercontent.com/87482328/143949764-0cf9a12b-9f10-48b8-8f06-7ccd3a962d6d.png)

*en*: Figure 3 shows the vaccination rate x dates for the Northeast region of Brazil. The black dots mark the real data from the dataset, the red curve represents the result of the regression, while the blue curve is the prediction result for 7, 14, 21 and 28 days ahead.

*pt-br*: A Figura 3 apresenta o gráfico da taxa de vacinação em relação a data para a região Nordeste do Brasil. Os pontos pretos são os valores reais do banco de dados, a curva em vermelho representa a regressão e a curva azul é o resultado da predição para 7, 14, 21 e 28 dias no futuro.

#### Table 3

| Dias | Taxa de vacinação (%) | Métricas |  Erro  |
| :--: | :-------------------: | :------: | :----: |
|  7   |         19.38         |   MAE    | 0.8097 |
|  14  |         20.22         |   RMSE   | 0.9650 |
|  21  |         21.02         |          |        |
|  28  |         21.79         |          |        |

*en*: Table 3 shows the forecasting results for vaccination rate on the Northeast region of Brazil.

*pt-br*: A tabela 3 mostra os resultados da predição da taxa de vacinação na região Nordeste do Brasil.

#### Figure 4

![image9](https://user-images.githubusercontent.com/87482328/143949765-2f7b9857-2b13-402c-b3ae-10ea00d4bc0d.png)

*en*: Figure 4 shows the vaccination rate x dates for the Midwest region of Brazil. The black dots mark the real data from the dataset, the red curve represents the result of the regression, while the blue curve is the prediction result for 7, 14, 21 and 28 days ahead.

*pt-br*: A Figura 4 apresenta o gráfico da taxa de vacinação em relação a data para a região Centro-Oeste do Brasil. Os pontos pretos são os valores reais do banco de dados, a curva em vermelho representa a regressão e a curva azul é o resultado da predição para 7, 14, 21 e 28 dias no futuro.

#### Table 4

| Dias | Taxa de vacinação (%) | Métricas |  Erro  |
| :--: | :-------------------: | :------: | :----: |
|  7   |         25.19         |   MAE    | 0.9801 |
|  14  |         28.17         |   RMSE   | 1.1901 |
|  21  |         31.44         |          |        |
|  28  |         34.93         |          |        |

*en*: Table 4 shows the forecasting results for vaccination rate on the Midwest region of Brazil.

*pt-br*: A tabela 4 mostra os resultados da predição da taxa de vacinação na região Centro-Oeste do Brasil.

#### Figure 5

![image11](https://user-images.githubusercontent.com/87482328/143949768-ecc21063-b339-40af-a569-4cb4f13b6c37.png)

*en*: Figure 5 shows the vaccination rate x dates for the Southeast region of Brazil. The black dots mark the real data from the dataset, the red curve represents the result of the regression, while the blue curve is the prediction result for 7, 14, 21 and 28 days ahead.

*pt-br*: A Figura 5 apresenta o gráfico da taxa de vacinação em relação a data para a região Sudeste do Brasil. Os pontos pretos são os valores reais do banco de dados, a curva em vermelho representa a regressão e a curva azul é o resultado da predição para 7, 14, 21 e 28 dias no futuro.

#### Table 5

| Dias | Taxa de vacinação (%) | Métricas |  Erro  |
| :--: | :-------------------: | :------: | :----: |
|  7   |         21.68         |   MAE    | 0.7834 |
|  14  |         22.66         |   RMSE   | 0.9804 |
|  21  |         23.58         |          |        |
|  28  |         24.40         |          |        |

*en*: Table 5 shows the forecasting results for vaccination rate on the Southeast region of Brazil.

*pt-br*: A tabela 5 mostra os resultados da predição da taxa de vacinação na região Sudeste do Brasil.

#### Figure 6

![image12](https://user-images.githubusercontent.com/87482328/143949770-77ebc40b-77e2-40f7-b407-36aa81be5a9d.png)

*en*: Figure 6 shows the vaccination rate x dates for the South region of Brazil. The black dots mark the real data from the dataset, the red curve represents the result of the regression, while the blue curve is the prediction result for 7, 14, 21 and 28 days ahead.

*pt-br*: A Figura 6 apresenta o gráfico da taxa de vacinação em relação a data para a região Sul do Brasil. Os pontos pretos são os valores reais do banco de dados, a curva em vermelho representa a regressão e a curva azul é o resultado da predição para 7, 14, 21 e 28 dias no futuro.

#### Table 6

| Dias | Taxa de vacinação (%) | Métricas |  Erro  |
| :--: | :-------------------: | :------: | :----: |
|  7   |         25.00         |   MAE    | 0.8019 |
|  14  |         28.29         |   RMSE   | 0.9565 |
|  21  |         32.18         |          |        |
|  28  |         36.61         |          |        |

*en*: Table 6 shows the forecasting results for vaccination rate on the South region of Brazil.

*pt-br*: A tabela 6 mostra os resultados da predição da taxa de vacinação na região Sul do Brasil.

---

## Paper

SÁ, Gabriel Caldas Barros e et al. (2021) [Predição Da Taxa de Ocupação de Leitos de UTI Por COVID-19 No Brasil Usando SVR](https://www.even3.com.br/anais/cobicet/374955-predicao-da-taxa-de-ocupacao-de-leitos-de-uti-por-covid-19-no-brasil-usando-svr/).. In: Anais do Congresso Brasileiro Interdisciplinar em Ciência e Tecnologia. Anais...Diamantina(MG) UFVJM.



