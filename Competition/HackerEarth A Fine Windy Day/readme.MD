### My Approach in a nutshell:

1. The rows with null targets were removed.
2. train.csv was splited into train and test set with a ratio of 99:1.
3. For dealing with NaN of categorical features, a function named 'impute_categorical' was uses which:
	- added new column and replaced, if category is null then 1 else 0
	- took most occured category in that variable (mode)
	- replaced NaN values with most occured category in actual variable
4. The Nan values of numerical features were replaced with the mean of the corresponding columns.
5. Distribution and Outliers were check (Didn't help much to get better result. So skipped it into the given notebook).
6. Found most important features with:
	- Pearson's correlation coefficient and p-value for numerical features
	- Boxplot for categorical features
7. The categorical variables were passed for One Hot Encoding.
8. Dataset was scaled using RobustScaler.
9. XGBoost Regressior was used for predicting 'windmill_generated_power(kW/h)'.
10. Using Hyperparameter Tuning with Grid Search, the best parameters were passed inside the regressor(Tuning code is given in the 2nd notebook).
11. Accuracy was measured and the same process was done with test.csv file to predict and submit the results.
12. The entire process was done using the most important features achieved from step-6, but the result was not better than the previous one. So submitted the previous one as the best.