# import libs
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def age_grouping(age):
	if age < 0:
		return 'Unknown'
	elif 0 < age < 5:
		return 'Baby'
	elif 5 < age < 12:
		return 'Child'
	elif 12 < age < 18:
		return 'Teenager'
	elif 18 < age < 30:
		return 'Young Adult'
	elif 30 < age < 65:
		return 'Adult'
	elif 60 < age < np.inf:
		return 'Senior'
	else:
		return 'error'

def age_mapping(age_map):
	# map each Age value to a numerical value
	age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3,
				   'Student': 4, 'Young Adult': 5, 'Adult': 6,
				   'Senior': 7}
	return age_mapping['age_map']

def age_grouping_df(age):
	age_val = [-1, 0, 5, 12, 18, 30, 60, np.inf]
	age_label = ['Unknown', 'Baby', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']

	return pd.cut(age, bins=age_val, labels=age_label)

def age_mapping_df(age_map):
	# map each Age value to a numerical value
	age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3,
				   'Student': 4, 'Young Adult': 5, 'Adult': 6,
				   'Senior': 7}
	return age_map.map(age_mapping)

# def age_fill(age_map):
# 	age_title_mapping = {1: "Young Adult", 2: "Student",
# 						 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}
#
# 	if age_map == "Unknown":
# 		age_map = age_title_mapping[age_map]
#
# 	return age_map

def title_mapping_df(title):
	# map each of the title groups to a numerical value
	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3,
					 "Master": 4, "Royal": 5, "Rare": 6}
	return title.map(title_mapping)

def title_mapping(title):
	# map each of the title groups to a numerical value
	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3,
					 "Master": 4, "Royal": 5, "Rare": 6}
	return title_mapping[title]

def sex_mapping_df(sex):
	sex_mapping = {"male": 0, "female": 1}
	return sex.map(sex_mapping)

def sex_mapping(sex):
	sex_mapping = {"male": 0, "female": 1}
	return sex_mapping[sex]

def embarked_mapping_df(emb):
	embarked_mapping = {"S": 1, "C": 2, "Q": 3}
	return emb.map(embarked_mapping)

def embarked_mapping(emb):
	embarked_mapping = {"S": 1, "C": 2, "Q": 3}
	return embarked_mapping[emb]

def extraction(train, test):
	# not needed - SibSp,Parch,Ticket,Fare,Cabin
	# need - Pclass,Name,Sex,Age,Embarked, title from name

	# read files
	plt.style.use('fivethirtyeight')
	warnings.filterwarnings('ignore')

	# removing un-nessasary data
	train = train.drop(['Cabin'], axis=1)
	test = test.drop(['Cabin'], axis=1)

	train = train.drop(['Parch'], axis=1)
	test = test.drop(['Parch'], axis=1)

	train = train.drop(['Ticket'], axis=1)
	test = test.drop(['Ticket'], axis=1)
	# grouping
	# title from name
	combine = [train, test]
	for data in combine:
		data['Title'] = data.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)

	pd.crosstab(train['Title'], train['Sex'])
	for data in combine:
		data['Title'] = data['Title'].replace(['Lady', 'Capt', 'Col',
											   'Don', 'Dr', 'Major',
											   'Rev', 'Jonkheer', 'Dona'],
											  'Rare')

		data['Title'] = data['Title'].replace(
			['Countess', 'Lady', 'Sir'], 'Royal')
		data['Title'] = data['Title'].replace('Mlle', 'Miss')
		data['Title'] = data['Title'].replace('Ms', 'Miss')
		data['Title'] = data['Title'].replace('Mme', 'Mrs')

	train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

	# map each of the title groups to a numerical value
	for data in combine:
		data['Title'] = title_mapping_df(data['Title'])
		data['Title'] = data['Title'].fillna(0)

	# mr_age = train[train["Title"] == 1]["AgeGroup"].mode()  # Young Adult
	# miss_age = train[train["Title"] == 2]["AgeGroup"].mode()  # Student
	# mrs_age = train[train["Title"] == 3]["AgeGroup"].mode()  # Adult
	# master_age = train[train["Title"] == 4]["AgeGroup"].mode()  # Baby
	# royal_age = train[train["Title"] == 5]["AgeGroup"].mode()  # Adult
	# rare_age = train[train["Title"] == 6]["AgeGroup"].mode()  # Adult

	# age - group
	train["Age"] = train["Age"].fillna(-1)
	test["Age"] = test["Age"].fillna(-1)

	train['AgeGroup'] = age_grouping_df(train['Age'])
	test['AgeGroup'] = age_grouping_df(test['Age'])

	# print("Hre here .....")
	# train['AgeGroup1'].equals(train['AgeGroup'])
	age_title_mapping = {1: "Young Adult", 2: "Student",
						 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

	for x in range(len(train["AgeGroup"])):
		if train["AgeGroup"][x] == "Unknown":
			train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]

	for x in range(len(test["AgeGroup"])):
		if test["AgeGroup"][x] == "Unknown":
			test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]

	# map each Age value to a numerical value
	train['AgeGroup'] = age_mapping_df(train['AgeGroup'])
	test['AgeGroup'] = age_mapping_df(test['AgeGroup'])

	train.head()

	# dropping the Age, Name feature
	train = train.drop(['Age'], axis=1)
	test = test.drop(['Age'], axis=1)

	train = train.drop(['Name'], axis=1)
	test = test.drop(['Name'], axis=1)

	train['Sex'] = sex_mapping_df(train['Sex'])
	test['Sex'] = sex_mapping_df(test['Sex'])

	train['Embarked'] = embarked_mapping_df(train['Embarked'])
	test['Embarked'] = embarked_mapping_df(train['Embarked'])

	return train, test;
