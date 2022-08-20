# neural-network-for-ODE

Inspired by the method mentioned in paper [1] and [2], I tested here two examples, the simple pendulum and the revisited JMAK model.

In both examples, I generate firstly the datasets via code data.py und save them in folder "output". result.py shows how the whole error changes with different hyperparameters.

In folder "Example 1", pendulum_withactfuction.py and pendulum_withoutactfuction.py have different network structures. And pendulum_withoutactfuction_withpred.py has some additional code to do predictions beyong the dataset.

# References:
[1]Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. Neural Ordinary Differential Equations. 2019

[2]Barinder Thind. An Introduction to Neural Differential Equations

[3]Mikhail Surtsukov. https://github.com/msurtsukov/neural-ode/blob/master/Neural%20ODEs.ipynb
