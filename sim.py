import matplotlib.pyplot as plt
from Origin.coordinate_origin import inverse_matrix, tranformation_matrix
from numpy import pi, deg2rad, linspace, ones_like
import numpy as np

# theta_min = deg2rad(0)
# theta_max = deg2rad(180)
# theta = linspace(theta_min, theta_max, 300)
# r = ones_like(theta)*1


theta1, theta2 = inverse_matrix(17.5, 37.5)
theta1 = deg2rad(theta1)
theta2 = deg2rad(theta2)
print(np.rad2deg(theta1),"\n",np.rad2deg(theta2))


a1 = tranformation_matrix(theta1,0)
a2 = tranformation_matrix(theta2,1)
H0_2 = a1[0] @ a2[0]


# print(H0_2)
x_co_link1 = [0,a1[1]]
y_co_link1 = [0,a1[2]]
x_co_link2 = [a1[1],H0_2[0,3]]
y_co_link2 = [a1[2],H0_2[1,3]] 

plt.subplot()
plt.plot(x_co_link1,y_co_link1, '-r', label='link1')
plt.plot(x_co_link2,y_co_link2, '-b', label='link2')


plt.show()

