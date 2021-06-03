import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def f(x):
    global num
    num += 1
    return 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2

def l_r(x0, h):
    x1 = [x0[0] - h, x0[1]]
    df = []
    df.append((f(x0) - f(x1))/h)
    x2 = [x0[0], x0[1] - h]
    df.append((f(x0) - f(x2))/h)
    return df

def c_r(x0, h):
    x1 = [x0[0] + h, x0[1]]
    x11 = [x0[0] - h, x0[1]]
    df = []
    df.append((f(x1) - f(x11))/(2*h))
    x2 = [x0[0], x0[1] + h]
    x22 = [x0[0], x0[1] - h]
    df.append((f(x2) - f(x22))/(2*h))
    return df

def r_r(x0, h):
    x1 = [x0[0] + h, x0[1]]
    df = []
    df.append((f(x1) - f(x0))/h)
    x2 = [x0[0], x0[1] + h]
    df.append((f(x2) - f(x0))/h)
    return df

def hes(x0, h):
    r = []
    f2x1 = (f([x0[0] + 2* h, x0[1]]) - 2* f(x0) + f([x0[0] - 2* h, x0[1]]))/(4 * h**2)
    f2x1x2 = (f([x0[0] + h, x0[1] + h]) - f([x0[0] + h, x0[1] - h]) - f([x0[0] - h, x0[1] + h]) + f([x0[0] - h, x0[1] - h]))/(4 * h**2)
    f2x2 = (f([x0[0], x0[1] + 2* h]) - 2* f(x0) + f([x0[0], x0[1] - 2* h]))/(4 * h**2)
    r.append([f2x1, f2x1x2])
    r.append([f2x1x2, f2x2])
    return r
    


def newton(x0, h, eps):
    x = [x0]
    global x1_arr
    x1_arr = [x0[0]]
    global y1_arr
    y1_arr = [x0[1]]
    global z_arr
    z_arr = [f(x0)]
    k = 0
    while np.linalg.norm(l_r(x[-1], h)) >= eps:
        hes_inv = np.linalg.inv(hes(x[-1], h))
        s = hes_inv.dot(l_r(x[-1], h))
        d = np.matrix(l_r(x[-1], h))
        l = -(d.dot(s))/((s.dot(hes(x[-1],h))).dot(s.transpose()))
        l = float(l[0])
        x1 = x[-1] - s 
        x.append(x1)
        k += 1
    print(x1, k)

def newton1(x0, h, eps):
    global x1_arr
    x1_arr = [x0[0]]
    global y1_arr
    y1_arr = [x0[1]]
    global z_arr
    z_arr = [f(x0)]
    x = [x0]
    k = 0
    while True:
        hes_inv = np.linalg.inv(hes(x[-1], h))
        s = hes_inv.dot(r_r(x[-1], h))
        d = np.matrix(r_r(x[-1], h))
        l = -(d.dot(s))/((s.dot(hes(x[-1],h))).dot(s.transpose()))
        l = float(l[0])
        x1 = x[-1] - s
        x.append(x1)
        k += 1
        x1_arr.append(x1[0])
        y1_arr.append(x1[1])
        z_arr.append(f(x1))
        if np.linalg.norm(x[-1] - x[-2])/np.linalg.norm(x[-1]) <= eps and abs(f(x[-1]) - f(x[-2])) <= eps:
            break
    print(x1, k)
    
num = 0
x0 = [-1.2, 0]
y0 = f(x0)
S1 = [0, 1]
S2 = [1, 0]
eps = 10 ** (-3)
h = 10 ** (-5)
x1_arr = []
y1_arr = []
z_arr = []

#newton(x0, h, eps)
newton1(x0, h, eps)
print(num)



fig = plt.figure()
ax = fig.gca(projection='3d')


X = np.arange(-5, 5, 0.2)
Y = np.arange(-5, 5, 0.2)
X, Y = np.meshgrid(X, Y)
Z = ((1-X)**2) + (100 * ((Y-X**2)**2))


surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


ax.set_zlim(-50, 10000)


fig.colorbar(surf, shrink=0.5, aspect=5)


plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.plot(x1_arr, y1_arr, z_arr, 'r-')
plt.show()



