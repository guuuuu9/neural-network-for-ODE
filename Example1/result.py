import numpy as np
import matplotlib.pyplot as plt

# x = np.array([2,3,4,5,6])
# y = np.array([0.03178022429347038,0.031408101320266724,
# 0.032761890441179276,0.027190767228603363,0.034794822335243225])
# plt.ylim((0.01, 0.05))
# plt.plot(x, y, '*', ms=13)
# my_x_ticks = np.arange(2, 7, 1)
# plt.xticks(my_x_ticks, size = 13)
# my_y_ticks = np.arange(0.01, 0.06, 0.01)
# plt.yticks(my_y_ticks, size = 13)
# plt.xlabel('# inner neurons', fontdict={'size': 14})
# plt.ylabel('Errors', fontdict={'size': 14})
# plt.grid(True)
# plt.show()

# x = np.array([2,3,4,5,6])
# y = np.array([0.03578897938132286,0.06192513927817345,
# 0.07504072785377502,0.040210191160440445,0.10649055242538452])
# plt.ylim((0.01, 0.13))
# plt.plot(x, y, '*', ms=13)
# my_x_ticks = np.arange(2, 7, 1)
# plt.xticks(my_x_ticks, size = 13)
# my_y_ticks = np.arange(0.01, 0.13, 0.03)
# plt.yticks(my_y_ticks, size = 13)
# plt.xlabel('# inner neurons', fontdict={'size': 14})
# plt.ylabel('Errors', fontdict={'size': 14})
# plt.grid(True)
# plt.show()

x = np.array([2,3,4,5,6])
y = np.array([0.0324656143784523,0.03512466698884964,
0.03372492268681526,0.03281406685709953,0.034118421375751495])
plt.ylim((0.02, 0.05))
plt.plot(x, y, '*', ms=13)
my_x_ticks = np.arange(2, 7, 1)
plt.xticks(my_x_ticks, size = 13)
my_y_ticks = np.arange(0.02, 0.06, 0.01)
plt.yticks(my_y_ticks, size = 13)
plt.xlabel('# inner neurons', fontdict={'size': 14})
plt.ylabel('Errors', fontdict={'size': 14})
plt.grid(True)
plt.show()