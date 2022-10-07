# Logistic Regression Simple Algorithm

## Introduction
Logistic regression is a method to create a model by using binary data (0 and 1). The goal is to predict something independent variable based on dependent variable. In real application, logistic regression is applied to predict a number of customers who buy a product or who did not based on their previous transaction, to predict a number of fraud transactions in credit card, and so on. In logistic regression, Y-axis lies from 0 – 1.

## Explanation

Logistic regression cannot be solved by using linear equation like linear regression. That because if the Y-axis of logistic function is transformed to linear function, the boundary of Y-axis lies from -infinity to +infinity. Then, when we calculate the misfit error between actual data and predicted data, it will not get good misfit error. The logistic function (also called sigmoid function) that can be described through Equation 1

Equation 1

$s(z) = {1 \over 1 + \exp^{-z}}$

where the $z$ is the logistic regression as written in equation 2 below. It looks like a linear regression term.

Equation 2

$z = b_0 + b_1X_1 + b_2X_2 + ... + b_nX_n$

the $b_i$ is the weight coefficient where the $b_0$ is an intercept and $b_1 - b_n$ is the coefficients of logistic regression. The $X_n$ is a number of features in a dataset that has a M x N dimensions. The M and N are number of rows and columns respectively. The $b_0$ has 1 x N dimensions, where the N is number of columns. As we can see in the Equation 2, the $b_1 - b_n$ have the $X$ feature that belongs to each $b$, except the $b_0$. However, in algorithm term, the $b_0$ has the $X$ feature that the values are default to 1. Then, the Equation 2 can be written

Equation 3

$z = X . b^T$

so the output of $z$ has M x 1 domensions or M rows.

To assess the quality of logistic regression, we calculate a [Cost Function](https://www.geeksforgeeks.org/ml-cost-function-in-logistic-regression/) as written in Equation 4. 

Equation 4

$J(\theta) = -{1 \over m} \varSigma_{i=1}^{n} [y_i log(h_{\theta}(x_i)) + (1-y_i)log(1-h_{\theta}(x_i))]$

where the $h_\theta(x)$ is the hypothetcial of probability in logistic function that is calculated by using Equation 1. The cost function will be itterated to a number of iteration to get the lowest cost value. The lowest cost value indicate the weight coefficient in the proper portion. It means, the weight coefficient will be updated through the iteration process or called gradient descent. The gradient descent can be calculated by using Equation 5 below

Equation 5 

$g = {1 \over m} X^T (s(z) - y)$

then multiplied by learning rate ($\alpha = 1$)

$g = g * \alpha$

then update the new weight coefficient

$b = b - g$

## Example Case

We have a dataset of netwrok purchased by customers. The task is to identify would the customer Purchased the network based on their Gender, Age and Estimation Salary. The dataset is downloaded from [here](https://github.com/sawarni99/Simple-Logistic-Regression).

### Overview Dataset

The dataset have a number of features as follows

```
#   Column           Non-Null Count  Dtype
---  ------           --------------  -----
 0   Gender           400 non-null    object
 1   Age              400 non-null    int64
 2   EstimatedSalary  400 non-null    int64
 3   Purchased        400 non-null    int64
```

## References

1. https://github.com/sawarni99/Simple-Logistic-Regression
2. https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
3. https://medium.com/analytics-vidhya/derivative-of-log-loss-function-for-logistic-regression-9b832f025c2d

## Contact
:email: auliakhalqillah.mail@gmail.com
