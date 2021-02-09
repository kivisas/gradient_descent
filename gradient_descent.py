#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:14:18 2019

@author: Sasa Kivisaari
sasa.kivisaari@aalto.fi
"""
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt

#Get the Boston housing data from csv
data = genfromtxt('mass_boston.csv', delimiter=',', dtype=float, names=True)

#Select variables for inspection
bh_vars = ['crim', 'nox', 'tax', 'age']

#Prepare plots
fig, axs = plt.subplots(1, 4, figsize=(15,6),facecolor='w', edgecolor='k')
axs = axs.ravel()

###############################################
#Big loop for checking out different variables#
###############################################

for j, bh_var in enumerate(bh_vars):
    #Pick per a variable for x
    x =  data[bh_var]
    x /= np.max(x) #Normalize x
    #Pick median value of owner-occupied homes as y
    y = data['medv']
    y /= np.max(y) #normalize y
    
    ###############################
    ## Calculate gradient descent##
    ###############################
    
    w = np.random.uniform()
    b = np.random.uniform()
    #Specify parameters
    L = 0.01  # The learning rate
    iterations = 10000  # The number of iterations for gradient descent
    n = float(len(x)) # Number of observatons
    
    # Smaller loop for performing Gradient Descent 
    for i in range(iterations): 
        y_pred = w*x + b  # Current predicted value of y
        #Loss function (MSE)
        D_w = (-2/n) * sum(x * (y - y_pred))  # Derivative with respect to w
        D_b = (-2/n) * sum(y - y_pred)  # Derivative with respect to b
        w = w - L * D_w  # Update w
        b = b - L * D_b  # Update b
        
    print (w, b)
    
    #############################################
    #Calculate linear regression using polyfit # 
    ############################################
    
    beta, error = np.polyfit(x,y, deg=1)
    
    #############################################
    #Plot regression line from linear regression#
    #############################################
    
    axs[j].scatter(x,y, marker="+") #Scatter plot
    w_gd, = axs[j].plot(x, b + w * x, '-', 
               label='Gradient descent')  #Gradient descent
    beta, = axs[j].plot(x, error + beta * x, '-', 
                    label='Linear regression') #Linear regression coefficient

    axs[j].set_xlabel(bh_var)
    axs[j].set_ylabel('medv')
    axs[j].legend([w_gd, beta], ['Gradient descent', 'Linear regression'])
    fig.savefig("plot.pdf", bbox_inches='tight')
