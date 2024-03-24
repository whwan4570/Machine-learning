import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy import array
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('./reData/House.csv')

print(df.columns)

inf_price = df.groupby('InflationRate').agg({'AveragePrice': np.mean}).reset_index()
sns.lmplot(data=inf_price, x='InflationRate', y='AveragePrice')
plt.show()

int_price = df.groupby('InterestRate').agg({'AveragePrice': np.mean}).reset_index()
sns.lmplot(data=int_price, x='InterestRate', y='AveragePrice')
plt.show()

int_price = df.groupby('MortgageRate').agg({'AveragePrice': np.mean}).reset_index()
sns.lmplot(data=int_price, x='MortgageRate', y='AveragePrice')
plt.show()

int_price = df.groupby('SupplyRate').agg({'AveragePrice': np.mean}).reset_index()
sns.lmplot(data=int_price, x='SupplyRate', y='AveragePrice')
plt.show()

int_price = df.groupby('UnemploymentRate').agg({'AveragePrice': np.mean}).reset_index()
sns.lmplot(data=int_price, x='UnemploymentRate', y='AveragePrice')
plt.show()

int_price = df.groupby('PopulationGrowth').agg({'AveragePrice': np.mean}).reset_index()
sns.lmplot(data=int_price, x='PopulationGrowth', y='AveragePrice')
plt.show()

start_date = '2005-01-01'
end_date = '2008-01-01'

mortgageSubprimeDF = df[(df['date'] >= start_date)&(df['date'] <= end_date)]

start_date1 = '2019-12-01'
end_date1 = '2023-02-01'

CoronaDF = df[(df['date'] >= start_date1)&(df['date'] <= end_date1)]

# print("---------------------------------")
# print(mortgageSubprimeDF.head(10))
# print("---------------------------------")
# print(CoronaDF.head(10))

data_cols = ['InflationRate', 'InterestRate', 'MortgageRate', 'SupplyRate', 'UnemploymentRate', 'PopulationGrowth']
label = 'AveragePrice'
features = ['InflationRate', 'InterestRate', 'MortgageRate', 'SupplyRate', 'UnemploymentRate', 'PopulationGrowth']
# Standardize the data
scaler = StandardScaler()
mortgageSubprimeScaled = scaler.fit_transform(mortgageSubprimeDF[data_cols])
coronaScaled = scaler.fit_transform(CoronaDF[data_cols])

# Perform PCA on the data
pca = PCA()
mortgageSubprimePCA = pca.fit_transform(mortgageSubprimeScaled)
coronaPCA = pca.fit_transform(coronaScaled)

def biplot(score,coeff,pcax,pcay,labels=None):
  pca1=pcax-1
  pca2=pcay-1
  xs = score[:,pca1]
  ys = score[:,pca2]
  n=score.shape[1]
  scalex = 1.0/(xs.max()- xs.min())
  scaley = 1.0/(ys.max()- ys.min())
  plt.scatter(xs*scalex,ys*scaley, c= mortgageSubprimeDF['AveragePrice'])
  for i in range(n):
    plt.arrow(0, 0, coeff[i,pca1], coeff[i,pca2],color='r',alpha=0.5)
    if labels is None:
      plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
    else:
      plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, labels[i], color='g', ha='center', va='center')
  plt.xlim(-1,1)
  plt.ylim(-1,1)
  plt.xlabel("PC{}".format(pcax))
  plt.ylabel("PC{}".format(pcay))
  plt.grid()


# Show 'Mortgage SubprimeDF PCA' plot
plt.figure(figsize=(10, 5))
plt.scatter(mortgageSubprimePCA[:, 0], mortgageSubprimePCA[:, 1], c=mortgageSubprimeDF['AveragePrice'])
plt.colorbar()
plt.title('Mortgage SubprimeDF PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

plt.title("Mortgage Subprime PCA")
biplot(mortgageSubprimePCA, pca.components_, 1, 2, labels=features)
plt.show()

def biplot1(score,coeff,pcax,pcay,labels=None):
  pca1=pcax-1
  pca2=pcay-1
  xs = score[:,pca1]
  ys = score[:,pca2]
  n=score.shape[1]
  scalex = 1.0/(xs.max()- xs.min())
  scaley = 1.0/(ys.max()- ys.min())
  plt.scatter(xs*scalex,ys*scaley, c= CoronaDF['AveragePrice'])
  for i in range(n):
    plt.arrow(0, 0, coeff[i,pca1], coeff[i,pca2],color='r',alpha=0.5)
    if labels is None:
      plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
    else:
      plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, labels[i], color='g', ha='center', va='center')
  plt.xlim(-1,1)
  plt.ylim(-1,1)
  plt.xlabel("PC{}".format(pcax))
  plt.ylabel("PC{}".format(pcay))
  plt.grid()

# Show 'CoronaDF PCA' plot
plt.figure(figsize=(10, 5))
plt.scatter(coronaPCA[:, 0], coronaPCA[:, 1], c=CoronaDF['AveragePrice'])
plt.colorbar()
plt.title('CoronaDF PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

plt.title("Corona PCA")
biplot1(coronaPCA, pca.components_, 1, 2, labels=features)
plt.show()
