
#import libs
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import data_extraction

#read files
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')

train1 = pd.read_csv('data/train.csv')
test1 = pd.read_csv('data/test.csv')

train, test = data_extraction.extraction(train1, test1)

def randon_forest(final):
	# prediction - randomforest
	from sklearn.model_selection import train_test_split

	# Drop the Survived and PassengerId
	# column from the trainset
	predictors = train.drop(['Survived', 'PassengerId'], axis=1)
	target = train["Survived"]
	x_train, x_val, y_train, y_val = train_test_split(
		predictors, target, test_size=0.33, random_state=0)

	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import accuracy_score

	randomforest = RandomForestClassifier()

	# Fit the training data along with its output
	randomforest.fit(x_train, y_train)

	y_pred = randomforest.predict(x_val)

	# Find the accuracy score of the model
	acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

	# result = randomforest.predict()
	result_val = randomforest.predict(final)

	return acc_randomforest, result_val

