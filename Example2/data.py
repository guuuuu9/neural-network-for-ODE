import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from scipy.integrate import odeint


# define the equations
def equations(z, t):
	w = z[0]
	x = z[1]
	y = z[2]
	P = z[3]
	dwdt = 1
	dxdt = 2.9*w
	dydt = 2*2.9*x
	dPdt = 4*math.pi*(1-P)*2.9*y
	dzdt = [dwdt, dxdt, dydt, dPdt]
	return dzdt

def plot_results(time, X):
	# plt.plot(time, X[:,0],label='w(t)')
	# plt.plot(time, X[:,1],label='x(t)')
	# plt.plot(time, X[:,2],label='y(t)')
	plt.plot(time, X[:,3],label='P(t)')
	plt.xlabel('time (s)',fontdict={'size': 14})
	plt.ylabel('values',fontdict={'size': 14})
	plt.xticks(size = 13)
	plt.yticks(size = 13)
	plt.legend(loc='best')
	plt.grid(True)
	plt.show()

z0 = [0,0,0,0]

# find the solution to the nonlinear problem
time = np.linspace(0,1,500)
X = odeint(equations, z0,  time)
plot_results(time, X)


# X = odeint(equations, z0,  time)
# data = { 't':time, 'x1': X[:,0], 'x2': X[:,1], 'x3': X[:,2], 'x4': X[:,3]}
# data = pd.DataFrame(data)
# data.to_excel("./output/data.xls", index=False)


# n = 301
# t = np.linspace(0,30,n)
# z0 = [0,0,0,0]

# w = np.empty_like(t)
# x = np.empty_like(t)
# y = np.empty_like(t)
# P = np.empty_like(t)
# w[0] = z0[0]
# x[0] = z0[1]
# y[0] = z0[2]
# P[0] = z0[3]

# for i in range(1,n):
# 	tspan = [t[i-1],t[i]]
# 	z = odeint(equations,z0,tspan)
# 	w[i] = z[1][0]
# 	x[i] = z[1][1]
# 	y[i] = z[1][2]
# 	P[i] = z[1][3]
# 	z0 = z[1]

# # plot results
# plt.plot(t,P,'r--',label='y(t)')
# plt.ylabel('values')
# plt.xlabel('time')
# plt.legend(loc='best')
# plt.show()
