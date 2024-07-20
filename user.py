import numpy as np
import data_extraction
import prediction

#data to predict
name = "rando"
pclass = 1
sex = 'male'					#"male": 0, "female": 1
sibsp = 1
fare = 71.06
embarked = 'S'
title = 'Mr'
age = 23
model = "random forest"

sex = data_extraction.sex_mapping(sex)
embarked = data_extraction.embarked_mapping(embarked)

title = data_extraction.title_mapping(title)

final = np.array([[pclass ,sex ,sibsp ,fare ,embarked ,title ,age]])

acc, result_val = prediction.randon_forest(final)

print("accuracy of the model is :",acc)
if result_val == 0:
	result = "Survived"
elif result_val == 1:
	result = "die"

print(name, " would ", result)