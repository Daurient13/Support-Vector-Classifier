
# Support Vector Classifier

In this project I will do a classification using the Support Vector Machine Algorithm. In machine learning, support-vector machines (SVMs, also support-vector networks) are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis. SVMs are one of the most robust prediction methods, being based on statistical learning frameworks. The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points. To separate the two classes of data points, there are many possible hyperplanes that could be chosen. Our objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.

# Dataset
The dataset I use is bank credit card data. There is not much explanation about the features in the data, because actually the data has been done by PCA (Principal Component Analysis), which I will do in Unsupervised Learning.

And this project I will do a classification whether this is a fraud or not

Data features consist of:

Time

V1 to V2

Amount

Class : 0 = Not fraud, 1 = Fraud

# Explanation
# Import Package
import common package

# Import Dataset
which i have explained before.

I always work on data science projects with simple think so that I can benchmark. Using a simple model to benchmark. And most of the time it's more efficient and sometimes find a good one. but at the beginning I did mini Exploratory Data Analysis. because i focus more on the algorithm

I didn't find the missing value in the data, because usually the data in the bank is completely tidy.

# Dataset Splitting
split the data into train, and test

X = all columns except the target column.

y = 'Class' as target

test_size = 0.2 (which means 80% for train, and 20% for test)

# Training
In the Training step there are 3 main things that I specify.

First, the preprocessor: here the columns will be grouped into numeric and categoric.
but if you look at the data there is no category column.

So i just use numeric column, which consists of 'Time', 'V1 to V28', 'Amount', and 'Class'

I also added standard scaling because SVM really helps with that

second, pipeline: contains the preprocessor as 'prep' which I defined earlier, and the algorithm as 'algo' which in this case I use Support Vector Classifier(SVC).

and third, tuning with Grid Search: in this case I use the tuning recommendations (gsp.svm_params) that often occur in many cases. but does not rule out hyperparameter tuning if the model results are not good. with cross validation = 3.

# Results
the test score reached 99%. remember every time we get a score that high, then we have to be suspicious and recheck:

1. Is there a data leak?
2. Is the dataset imbalanced?
3. is the problem something simple

In this case, what actually happened was an imbalanced dataset
So what is an imbalanced dataset? Let's check Class value_count, here it says that:

0 (Not Fraud)  = 30000

1 (Fraud)      = 492

it says, there are 30000 peoples who not make a Fraud, and 492 peoples who make a Fraud

it means that if we make a baseline, assume that people do not commit fraud as much as 30000 if divided by the total data of 30492

30000 / 30492 = 0.98, so we will get a score of 98% without doing anything, its because the baseline is very high due to imbalanced dataset

should be in a case like this Accuracy can't be used because it will deceive. if so, is there any other scoring besides accuracy?

Yes, we will use F1-Score, After we run it again, it turns out that the F1-Score is 97 and this is pretty good, and already using the F1-Scoring which has considered the data imbalance.

are we sure that this model is good? we will show it with Confusion Matrix.

The way to read the Confusion Matrix is ​​that we plot the actual vs prediction
