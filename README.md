# Predicting ICU bed occupancy rate by COVID-19 in Brazil and Regions using SVR based on vaccination.

This study sought to apply the SVR technique to predict the ICU bed occupancy rate by COVID-19 in Brazil for 7, 14, 21 and 28 days after May 10, 2021. See the document [here](https://www.even3.com.br/anais/cobicet/374955-predicao-da-taxa-de-ocupacao-de-leitos-de-uti-por-covid-19-no-brasil-usando-svr/).

## Data

- Data Source:  https://bigdata-covid19.icict.fiocruz.br/

- Number of instances:  27 instances

- Date of last instance:  10/05/2021 (day / month / year)

- Final data:

  ![Figure](https://github.com/Vinicius999/ICU-beds-forecast-covid-19/blob/main/images/dataset-image.png)

## Methodology
- Training data:  85%
- Validation data:  15%
- Metric:  MAE (Mean Absolute Error)
- No. of training and validation tests:  10
- Prediction time intervals:  7, 14, 21 and 28 days after the last collection date

## Results

### Training and validation

Figures 1, 2, 3, 4, 5, 6, 7, 8, 9 and 10 show the graph referring to the training and validation in each testing testing phase, respectively.

![Figure](https://github.com/Vinicius999/ICU-beds-forecast-covid-19/blob/main/images/tests-01-02-05-06.png)

![Figure](https://github.com/Vinicius999/ICU-beds-forecast-covid-19/blob/main/images/tests-03-04-07-08.png)

![Figure](https://github.com/Vinicius999/ICU-beds-forecast-covid-19/blob/main/images/tests-09-10.png)

The table shows the parameters that were changed in each test, as well as the respective MAE results. The parameters `gamma` and `coef0` were constant for all tests, with the values `'auto'` and `1`, respectively.

![Figure](https://github.com/Vinicius999/ICU-beds-forecast-covid-19/blob/main/images/tests-parameters-image.png)

### Prediction

In the testing phase, the parameters used in the 6th test were chosen because of the lowest MAE value obtained (8.80%), so the SVR function was as follows:

`SVR(kernel='poly', C=1, gamma='auto', degree=8, epsilon=0.1, coef0=1)`

The figure below shows the result of this prediction:

![Figure](https://github.com/Vinicius999/ICU-beds-forecast-covid-19/blob/main/images/predict-image.png)

- red dots: actual occupancy rates already available in the dataset;
- blue curve: regression for the already known values
- red curve: prediction of future days

## Paper

SÁ, Gabriel Caldas Barros e et al. (2021) [Predição Da Taxa de Ocupação de Leitos de UTI Por COVID-19 No Brasil Usando SVR](https://www.even3.com.br/anais/cobicet/374955-predicao-da-taxa-de-ocupacao-de-leitos-de-uti-por-covid-19-no-brasil-usando-svr/).. In: Anais do Congresso Brasileiro Interdisciplinar em Ciência e Tecnologia. Anais...Diamantina(MG) UFVJM.



