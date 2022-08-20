import numpy as np 
from numpy import sin, cos
import pandas as pd
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import xlwt


# define the equations
def equations(x0, t):
	theta, theta_dot = x0
	f = [theta_dot, -(g/l) * sin(theta)]
	return f

# parameters
g = 9.81
l = 1.0

time = np.arange(0, 30.0, 0.1)

# for i in range(0, 181, 1):
# 	initial_angle = i
# 	theta0i = np.radians(initial_angle)
# 	theta_dot0i = np.radians(0.0)
# 	Xi = odeint(equations, [theta0i, theta_dot0i],  time)
# 	datai = { 't':time, 'x1': Xi[:,0], 'x2': Xi[:,1]}
# 	datai = pd.DataFrame(datai)
# 	datai.to_excel("./output/data{}.xls".format(i), index=False)


def plot_results(time, X1, initial_angle):
	# plt.plot(X1[:,0], X1[:,1])
	plt.plot(time, X1[:,0])

	s = '(Initial Angle = ' + str(initial_angle) + ' degrees)'

	plt.title('Pendulum Motion'+s, fontdict={'size': 14})
	plt.xlabel('time (s)', fontdict={'size': 14})
	plt.ylabel(r'$\theta$ (rad)', fontdict={'size': 14})
	plt.xticks(size = 13)
	plt.yticks(size = 13)
	plt.grid(True)
	plt.show()

# initial conditions
initial_angle = 100.0
theta = np.radians(initial_angle)
theta_dot = np.radians(0.0)

# find the solution to the nonlinear problem
X1 = odeint(equations, [theta, theta_dot],  time)

plot_results(time, X1, initial_angle)