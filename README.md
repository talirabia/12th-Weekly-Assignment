## Overview of the Analysis

The importance of measuring risk when making financial decisions. We used risk metrics and analytics to make decisions about the market, investments, and even retirement portfolios. Financial risk is a serious concern in other areas of finance, too, such as credit, lending, and insurance. Because many factors relate to risk, we need better tools to consider various components for making predictions about the future. In fact, fintech companies have moved beyond traditional risk analytics and are starting to use machine learning to model and predict risk.
In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
he field of machine learning called supervised learning. Supervised learning trains an algorithm to learn based on a labeled dataset, where each item in the dataset is tagged with the answer. This provides an answer key that you can use to evaluate the accuracy of the training data. Supervised learning can consider multiple factors and known past outcomes to make predictions about future outcomes, such as financial risk.

* Explain what financial information the data was on, and what you needed to predict.
Compare and contrast regression and classification.Describe the model-fit-predict pattern that is used to create, train, and use machine learning models.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
A value of 0  means an healthy firm, and a value of 1 means a unhealthy firm. We can thus use the value_counts function to count the number of firms in each category.
* Describe the stages of the machine learning process you went through as part of this analysis.
often broadly divide machine learning into supervised learning, unsupervised learning, and reinforcement learning, we can further divide supervised learning into two types of algorithms: regression and classification.Regardless of whether we use a regression or a classification model, we create most supervised learning models by following a basic pattern: model-fit-predict. In this three-stage pattern, we present a machine learning algorithm with data (the model stage), and the algorithm learns from this data (the fit stage) to form a predictive model (the predict stage). A predictive model is simply the resulting model, where the algorithm has mathematically adjusted itself so that it can translate a new set of inputs to the correct output.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

Let's summarize the steps that we took to use a logistic regression model:

Create a model with LogisticRegression().

Train the model with model.fit().

Make predictions with model.predict().

Evaluate the model with accuracy_score(). Now, it’s your turn to apply what you learned about the preceding steps to a dataset that involves financial payments. The goal is to predict which payments are fraudulent and which aren’t.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

1. use the accuracy_score function to estimate the accuracy of the model. 
2. The precision metric relates to the accuracy metric but slightly differs. As with the accuracy, one way to illustrate the concept of precision is through a confusion matrix.
3. Another way that we can assess a model's performance is by using the recall, which is also called the sensitivity. (People in machine learning more commonly use the term recall.) In our example, the recall measures the number of actually fraudulent transactions that the model correctly classified as fraudulent.
* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
  
precision    recall  f1-score   support

           0       0.97      0.97      0.97     56294
           1       0.04      0.04      0.04      1858

    accuracy                           0.94     58152
   macro avg       0.50      0.50      0.50     58152
weighted avg       0.94      0.94      0.94     58152


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

precision    recall  f1-score   support

           0       1.00      1.00      1.00     18742
           1       0.87      0.91      0.89       642

    accuracy                           0.99     19384
   macro avg       0.93      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384
## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
The training data performs the best as compare to testing data. 
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
Looking at the two classification reports for the  test data, it looks as if model performance declined--albeit slightly--on the test data. This is to be expected: this is how well the model is performing on data that the model hasn't seen before. If we're still getting strong precision and recall on the test dataset, this is a good indication about how well the model is likely to perform in real life.

If you do not recommend any of the models, please justify your reasoning.
Training and here is why- Looking at the two classification reports for the  test data, it looks as if model performance declined--albeit slightly--on the test data. This is to be expected: this is how well the model is performing on data that the model hasn't seen before. If we're still getting strong precision and recall on the test dataset, this is a good indication about how well the model is likely to perform in real life.