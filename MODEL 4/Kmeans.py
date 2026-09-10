import pandas as pd
from sklearn.cluster import k_means
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt



data = pd.read_csv("income.csv")

print(data)


plt.scatter(data['Age'],data['Income($)'])
plt.show()

km = k_means(n_clusters=3)
print(km)