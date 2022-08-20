import numpy as np
import matplotlib.pyplot as plt

x = np.array([6,7,8,9,10])
y = np.array([0.0036151930689811707,0.000728901126421988,
0.0001863733195932582,0.003540163394063711,0.002544023562222719])
plt.ylim((0.000, 0.004))
plt.plot(x, y, '*', ms=13)
my_x_ticks = np.arange(6, 11, 1)
plt.xticks(my_x_ticks, size = 13)
my_y_ticks = np.arange(0.000, 0.005, 0.001)
plt.yticks(my_y_ticks, size = 13)
plt.xlabel('# inner neurons', fontdict={'size': 14})
plt.ylabel('Errors', fontdict={'size': 14})
plt.grid(True)
plt.show()


# x = np.array([800,1000,2000])
# y = np.array([0.001168619142845273,0.0010601901449263096,
# 0.0001863733195932582])
# plt.ylim([0.000, 0.0015])
# plt.plot(x, y, linewidth=2,  marker='o', ms = 6)
# my_x_ticks = np.arange(800, 2000, 500)
# plt.xticks(my_x_ticks, size = 13)
# my_y_ticks = np.arange(0.000, 0.002, 0.0005)
# plt.yticks(my_y_ticks, size = 13)
# plt.xlabel('# iterations', fontdict={'size': 14})
# plt.ylabel('Errors', fontdict={'size': 14})
# plt.grid(True)
# plt.show()

# 800
# 0.001168619142845273

# 1000
# 0.0010601901449263096

# 2000
# 0.0001863733195932582


# 0.01
# 0.5301001667976379

# 0.1
# 0.29590651392936707

# 1
# 0.05747972056269646

# 10
# 0.05736803263425827

# 100
# 0.001168619142845273

# 1000
# 3.19475371629384e+33



# 50
# 0.001168619142845273  0.0001863733195932582

# 45
# 0.0005423328257165849  0.0023571995552629232

# 40
# 0.0021451504435390234

# 55
# 0.011219508945941925   0.009873968549072742