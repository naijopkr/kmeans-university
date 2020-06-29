import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')

df = pd.read_csv('./data/college_data.csv')
df.head()
df.info()
df.describe()

# EDA
sns.scatterplot(x='Room.Board', y='Grad.Rate', hue='Private',data=df)
sns.scatterplot(x='Outstate', y='F.Undergrad', hue='Private',data=df)

def hist_private(feature = 'Outstate'):
    g = sns.FacetGrid(df, hue='Private', palette='coolwarm', size=6, aspect=2)
    g = g.map(plt.hist, feature, bins=20, alpha=0.7)
    plt.legend()

hist_private(feature='Outstate')
hist_private(feature='Grad.Rate')

# Name of the school with graduation rage higher 100%
df[df['Grad.Rate'] > 100] # Cazenovia College

# Set Grad.Rate for 100 to make sense
df.at[95, 'Grad.Rate'] = 100
df.at[95, 'Grad.Rate']
