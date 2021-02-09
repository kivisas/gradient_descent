Gradient descent excercise
=============
@Sasa Kivisaari (sasa.kivisaari@aalto.fi)

This script runs gradient descent and linear regression using Boston housing 
data set. In this excercise, I predict the median value of occupied homes 
('medv') with per capita crime rate ('crim'), nitrogen oxides concentration 
('nox'), full-value property-tax rate per $10,000. ('tax') and proportion of 
owner-occupied units built prior to 1940 ('age').


-------

**Gradient descent**

- Normalize x and y by dividing by maximum value to keep the algorithm 
  numerically stable. 
- Initially, assign numeric weight (w) and bias (b) random values from a 
  uniform distribution. 
- L is learning rate, which relates to how much w anb b can change at each 
  iteration. 
- Number of iterations is set to 10 000.
 
- The script calculates the the loss function (Mean squared error) with respect 
  to weight (w). 
- d_w is the value of the partial derivative with respect to weight (w). 
- d_b is the partial derivative with respect to bias (b).

-------

**Linear regression**
- Numpy.polyfit is used to predict y from x and estimate the coefficient (beta) 
  and residual (error). 

-------
**Plot**
- The values of x and y are indicated as a scatter plot. 
- The solutions of the gradient descent and linear regression, are demonstrated 
 with blue and orange regression lines, respectively. The plots are drawn 
 for different variables. 
- The similarity of of blue and orange regression lines indicates that the 
  gradient descent approximates linear regression. 
