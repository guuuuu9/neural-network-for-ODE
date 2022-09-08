import numpy as np
import matplotlib.pyplot as plt

x = np.array([6,7,8,9,10])
y = np.array([0.00048287719255313277,0.001072725048288703,
0.00045012799091637135,0.000639918667729944,0.0006788814207538962])
plt.ylim((0.000, 0.0012))
plt.plot(x, y, '*', ms=13)
my_x_ticks = np.arange(6, 11, 1)
plt.xticks(my_x_ticks, size = 13)
my_y_ticks = np.arange(0.000, 0.0014, 0.0002)
plt.yticks(my_y_ticks, size = 13)
plt.xlabel('# inner neurons', fontdict={'size': 14})
plt.ylabel('Errors', fontdict={'size': 14})
plt.grid(True)
plt.show()


# x = np.array([800,1200,2000])
# y = np.array([0.00563642056658864,0.0011156914988532662,
# 0.00045012799091637135])
# plt.ylim([0.000, 0.006])
# plt.plot(x, y, linewidth=2,  marker='o', ms = 6)
# my_x_ticks = np.arange(800, 2000, 500)
# plt.xticks(my_x_ticks, size = 13)
# my_y_ticks = np.arange(0.000, 0.008, 0.002)
# plt.yticks(my_y_ticks, size = 13)
# plt.xlabel('# iterations', fontdict={'size': 14})
# plt.ylabel('Errors', fontdict={'size': 14})
# plt.grid(True)
# plt.show()