"""@author: BANERJEE, Rohini (Student ID: 20543577)
MSBD5001: Foundations of Data Analytics
Title: InClass Kaggle Competition"""

# Import all necesssary libraries
import os
import pandas as pd
import numpy as np
import tensorflow as tf

# Read the train and test datasets
test = pd.read_csv("test.csv")
test.drop(['id'],axis=1,inplace=True) #Remove the 'id' column in test data
train = pd.read_csv("train1.csv")

# Replace all n_jobs=-1 with 16 (it is using all processors, which is probably 16 and rarely 32)
train.loc[train['n_jobs'] == -1, 'n_jobs'] = 16
test.loc[test['n_jobs'] == -1, 'n_jobs'] = 16

# Create few features for train data
train['n1'] = (train['n_classes'] * train['n_clusters_per_class'])/train['n_jobs']
train['n2'] = (train['max_iter'] * train['n_samples'])/train['n_jobs']
train['n3'] = train['n1']/train['n_informative']
# Create few features for test data
test['n1'] = (test['n_classes'] * test['n_clusters_per_class'])/test['n_jobs']
test['n2'] = (test['max_iter'] * test['n_samples'])/test['n_jobs']
test['n3'] = test['n1']/test['n_informative']

# Separate out the label from the train set
label = train["time"]
train.drop(['time'], axis=1, inplace=True)
# Separate out the categorical feature so that others can be scaled
pen_train = train["penalty"]
pen_test = test["penalty"]

# Normalizing the features wrt to the mean and standard deviation of test set
# This is because we want it to generalize well on the test set
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
train_new = train.select_dtypes(include=numerics)
test_new = test.select_dtypes(include=numerics)
train1 = (train_new - test_new.mean())/test_new.std(ddof=0)
train1 = train1.join(pen_train) # Merge the categorical feature after scaling is over
train1.drop(['l1_ratio','scale','random_state','alpha','flip_y'],axis=1, inplace=True)
# Normalizing test features as well
test1 = (test_new - test_new.mean())/test_new.std(ddof=0)
test1 = test1.join(pen_test)
test1.drop(['l1_ratio','scale','random_state','alpha','flip_y'], axis=1, inplace=True)

# Set up the tensorflow model
BATCH_SIZE = 128
num_epochs = 5000
input_train = tf.estimator.inputs.pandas_input_fn(x=train1,y=label,batch_size=BATCH_SIZE,num_epochs=num_epochs,shuffle=True)
# Set all input features to abide by tensorflow requirements
max_iter = tf.feature_column.numeric_column("max_iter")
n_jobs = tf.feature_column.numeric_column("n_jobs")
n_samples = tf.feature_column.numeric_column("n_samples")
n_features = tf.feature_column.numeric_column("n_features")
n_classes = tf.feature_column.numeric_column("n_classes")
n_clusters_per_class = tf.feature_column.numeric_column("n_clusters_per_class")
n_informative = tf.feature_column.numeric_column("n_informative")
n1 = tf.feature_column.numeric_column("n1")
n2 = tf.feature_column.numeric_column("n2")
n3 = tf.feature_column.numeric_column("n3")
penalty = tf.feature_column.categorical_column_with_vocabulary_list(key="penalty", vocabulary_list=["l2", "l1", "none", "elasticnet"])
# Declare all the feature columns
Feature_columns = [max_iter,  n_jobs,  n_samples,  n_features,
    n_classes,  n_clusters_per_class,  n_informative,  n1,  n2,
    n3,  tf.feature_column.indicator_column(penalty),]
# Declare which columns have to be taken as 'wide'
wide_columns = [ max_iter, n_jobs, n_samples, n_features,
    n_classes, n_clusters_per_class, n_informative, n1, n2, n3,]

# Train the tensorflow model
m = tf.estimator.DNNLinearCombinedRegressor(
    linear_feature_columns=wide_columns,
    dnn_feature_columns=Feature_columns,
    dnn_hidden_units=[1000, 500, 240, 150, 75, 25, 14],
    dnn_activation_fn=tf.nn.relu
    )
m.train(input_fn=input_train)
print('Training of the model is done!')

# Predict the output for test data
predict_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=test1,
        batch_size=1,
        num_epochs=1,
        shuffle=False)
predictions = m.predict(input_fn=predict_input_fn)
# Get the predicted results into a list
result = []
for i in predictions:
    result.append(i["predictions"][0])

# Save it to a CSV file in required format
test_id = np.arange(100)
test_id = test_id.reshape(len(test_id),1)
result = np.array(result)
result = result.reshape(len(result),1)
result = np.abs(result) # take absolute of any negatively predicted value
output = np.concatenate((test_id,result), axis=1)
np.savetxt("submission.csv", output, delimiter=",", fmt='%i,%f', header="Id,Time", comments='')
