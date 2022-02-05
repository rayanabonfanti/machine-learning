import pandas as pd
dados = pd.read_csv('consumocerveja.csv', sep=';')
dados.head()
dados.shape
dados.describe().round(2)
from sklearn.model_selection import train_test_split
X = dados[['temp_max', 'chuva', 'fds']]
y = dados['consumo']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2811)
from sklearn.linear_model import LinearRegression
from sklearn import metrics
modelo = LinearRegression()
modelo.fit(X_train, y_train)
print('R² = {}'.format(modelo.score(X_train, y_train).round(2)))
y_previsto = modelo.predict(X_test)
print('R² = %s' % metrics.r2_score(y_test, y_previsto).round(2))
temp_max=40
chuva=0
fds=1
entrada=[[temp_max, chuva, fds]]

print('{0:.2f} litros'.format(modelo.predict(entrada)[0]))
