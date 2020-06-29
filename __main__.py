import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')

df = pd.read_csv('./data/college_data.csv')
df.head()
df.info()
df.describe()


# EDA
def scatterplot(x='Room.Board', y='Grad.Rate', hue='Private'):
    sns.scatterplot(x=x, y=y, hue=hue, data=df)

def hist_private(feature = 'Outstate'):
    g = sns.FacetGrid(df, hue='Private', palette='coolwarm', size=6, aspect=2)
    g = g.map(plt.hist, feature, bins=20, alpha=0.7)
    plt.legend()


scatterplot(x='Room.Board', y='Grad.Rate', hue='Private')
scatterplot(x='Outstate', y='F.Undergrad', hue='Private')


hist_private(feature='Outstate')
hist_private(feature='Grad.Rate')


# Name of the school with graduation rage higher 100%
df[df['Grad.Rate'] > 100] # Cazenovia College

# Set Grad.Rate for 100 to make sense
df.at[95, 'Grad.Rate'] = 100
df.at[95, 'Grad.Rate']

# K Means Cluster Creation
from sklearn.cluster import KMeans

X = df.drop(['Unnamed: 0', 'Private'], axis=1)
X.head()

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Cluster center vectors
kmeans.cluster_centers_

# Evaluation
df['Cluster'] = kmeans.labels_
df.head()

df['Private'] = df['Private'].apply(lambda x: 1 if x == 'Yes' else 0)
df.head()

from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(df['Private'], df['Cluster'])
cr = classification_report(df['Private'], df['Cluster'])

print(cm)
print(cr)

scatterplot(hue='Cluster')
scatterplot(hue='Private')
scatterplot(x='Outstate', y='F.Undergrad', hue='Cluster')
scatterplot(x='Outstate', y='F.Undergrad', hue='Private')
