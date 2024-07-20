import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# To know number of columns and rows
# train.shape
# # (891, 12)
#
train.info()
train.isnull().sum()

#visualise

# survived
f, ax = plt.subplots(1, 2, figsize=(12, 4))
train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=False)
ax[0].set_title('Survivors (1) and the dead (0)')
ax[0].set_ylabel('')
sns.countplot(x='Survived', data=train, ax=ax[1])
ax[1].set_ylabel('Quantity')


# sex
f, ax = plt.subplots(1, 2, figsize=(12, 4))
train[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survivors by sex')
sns.countplot(x='Sex', hue='Survived', data=train, ax=ax[1])
ax[1].set_ylabel('Quantity')
ax[1].set_title('Survived (1) and deceased (0): men and women')
plt.show()
#

#pclass
# f, ax = plt.subplots(1, 3, figsize=(12, 4))
# train[['Pclass', 'Survived']].groupby(['Pclass']).mean().plot.bar(ax=ax[0])
# ax[0].set_title('Survivors by pclass')
# sns.countplot(x='Pclass', hue='Survived', data=train, ax=ax[1])
# ax[1].set_ylabel('Quantity')
# ax[1].set_title('Survived (1) and deceased (0): class')
# plt.show()
#

#embarked
# f, ax = plt.subplots(1, 3, figsize=(12, 4))
# train[['Embarked', 'Survived']].groupby(['Embarked']).mean().plot.bar(ax=ax[0])
# ax[0].set_title('Survivors by Embarked')
# sns.countplot(x='Embarked', hue='Survived', data=train, ax=ax[1])
# ax[1].set_ylabel('Quantity')
# ax[1].set_title('Survived (1) and deceased (0): class')
# plt.show()

