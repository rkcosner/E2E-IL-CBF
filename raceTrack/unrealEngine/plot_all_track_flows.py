
import matplotlib.pyplot as plt
import numpy as np
import os
from plotting_utils import * 
pi = np.pi


l_fulltrack = 20 
l = l_fulltrack/4
w = 2
TURN_CBF = True
turn_delta = 0.1

SAVE_RGB = True
OVERWRITE_DATA = False
END2END = True

if not END2END: 
    prepend = "trop"
else: 
    prepend = ""

def getCBFValue(state): 

    l = l_fulltrack/4
    x = state[0, 0]
    y = state[1, 0]
    theta = state[2, 0]
    car_heading = np.array([[np.cos(theta)], [np.sin(theta)]])


    r = l / pi

    bits = []

    g = np.array([
        [np.cos(theta), 0],
        [np.sin(theta), 0],
        [0, 1]
    ])

    # Stay in Outer Track
    Lfh = 0
    dhdx = np.zeros((1, 3))
    if x >= l / 2:
        h = (r + w) ** 2 - ((x - l / 2) ** 2 + y ** 2)
        dhdx[0, 0] = -2 * (x - l / 2)
        dhdx[0, 1] = -2 * y

        if TURN_CBF:
            phi = np.arctan2(y, x-l/2)
            d = np.array([[-np.cos(phi)], [-np.sin(phi)]])
            rad = (x -l/2)**2 +y**2
            dperp = np.array([[np.sin(phi)],[-np.cos(phi)]])
            dhdstate = turn_delta*(car_heading.T@dperp)*np.array([[-y/rad, (x-l/2)/rad]])
            dhdx[0, 0] += dhdstate[0,0]
            dhdx[0, 1] += dhdstate[0,1]
            dhdx[0, 1] += turn_delta*d.T@np.array([[-np.sin(theta)], [np.cos(theta)]])
            h += turn_delta*((car_heading.T@d)[0,0]-1)

    elif x <= -l / 2:
        h = (r + w) ** 2 - ((x + l / 2) ** 2 + y ** 2)
        dhdx[0, 0] = -2 * (x + l / 2)
        dhdx[0, 1] = -2 * y

        if TURN_CBF:
            phi = np.arctan2(y, x-l/2)
            d = np.array([[-np.cos(phi)], [-np.sin(phi)]])
            rad = (x + l/2)**2 +y**2
            dperp = np.array([[np.sin(phi)],[-np.cos(phi)]])
            dhdstate = turn_delta*(car_heading.T@dperp)*np.array([[-y/rad, (x+ l/2)/rad]])
            dhdx[0, 0] += dhdstate[0,0]
            dhdx[0, 1] += dhdstate[0,1]
            dhdx[0, 1] += turn_delta*d.T@np.array([[-np.sin(theta)], [np.cos(theta)]])
            h += turn_delta*((car_heading.T@d)[0,0]-1)
    elif y <= 0:
        h = (r + w)**2 - y**2
        dhdx[0, 1] = -2*y
        if TURN_CBF:
            d = np.array([[0], [1]])
            dhdx[0, 0] += 0
            dhdx[0, 1] += 0
            dhdx[0, 2] += turn_delta*d.T@np.array([[-np.sin(theta)], [np.cos(theta)]])
            h += turn_delta*((car_heading.T@d)[0,0]-1)
    elif y > 0:
        h = (r + w)**2 - y**2
        dhdx[0, 1] = -2*y
        if TURN_CBF:
            d = np.array([[0], [-1]])
            dhdx[0, 0] += 0
            dhdx[0, 1] += 0
            dhdx[0, 2] += turn_delta*d.T@np.array([[-np.sin(theta)], [np.cos(theta)]])
            h += turn_delta*((car_heading.T@d)[0,0]-1)

    Lgh = dhdx @ g

    bits.append([Lfh, Lgh, h])

    # Stay out of Inner Track
    dhdx = np.zeros((1, 3))
    Lfh = 0
    phi = 0
    if x >= l / 2:
        h = ((x - l / 2) ** 2 + y ** 2) - (r) ** 2
        dhdx[0, 0] = 2 * (x - l / 2)
        dhdx[0, 1] = 2 * y
        if TURN_CBF:
            phi = np.arctan2(y, x-l/2)
            d = np.array([[np.cos(phi)], [np.sin(phi)]])
            rad = (x -l/2)**2 +y**2
            dperp = np.array([[-np.sin(phi)],[np.cos(phi)]])
            dhdstate = turn_delta*(car_heading.T@dperp)*np.array([[-y/rad, (x-l/2)/rad]])
            dhdx[0, 0] += dhdstate[0,0]
            dhdx[0, 1] += dhdstate[0,1]
            dhdx[0, 1] += turn_delta*d.T@np.array([[-np.sin(theta)], [np.cos(theta)]])
            h += turn_delta*((car_heading.T@d)[0,0]-1)
    elif x <= -l / 2:
        h =  ((x + l / 2) ** 2 + y ** 2) - (r) ** 2
        dhdx[0, 0] = 2 * (x + l / 2)
        dhdx[0, 1] = 2 * y
        if TURN_CBF:
            phi = np.arctan2(y, x+l/2)
            d = np.array([[np.cos(phi)], [np.sin(phi)]])
            rad = (x +l/2)**2 +y**2
            dperp = np.array([[-np.sin(phi)],[np.cos(phi)]])
            dhdstate = turn_delta*(car_heading.T@dperp)*np.array([[-y/rad, (x+l/2)/rad]])
            dhdx[0, 0] += dhdstate[0,0]
            dhdx[0, 1] += dhdstate[0,1]
            dhdx[0, 2] += turn_delta*d.T@np.array([[-np.sin(theta)], [np.cos(theta)]])
            h += turn_delta*((car_heading.T@d)[0,0]-1)
    elif y <= 0:
        h = y**2 - r**2
        dhdx[0, 1] = 2*y
        if TURN_CBF:
            d = np.array([[0], [-1]])
            dhdx[0, 0] += 0
            dhdx[0, 1] += 0
            dhdx[0, 2] += turn_delta*d.T@np.array([[-np.sin(theta)], [np.cos(theta)]])
            h += turn_delta*((car_heading.T@d)[0,0]-1)
    elif y > 0:
        h = y**2 - r**2
        dhdx[0, 1] = 2*y
        if TURN_CBF:
            d = np.array([[0], [1]])
            dhdx[0, 0] += 0
            dhdx[0, 1] += 0
            dhdx[0, 2] += turn_delta*d.T@np.array([[-np.sin(theta)], [np.cos(theta)]])
            h += turn_delta*((car_heading.T@d)[0,0]-1)
    Lgh = dhdx @ g
    # if np.sum(np.abs(Lgh))<0.01:
    #     print("failure")
    #     print("phi = ", phi)
    #     print("d =", d)
    #     print("heading derivative = ", np.array([[-np.sin(theta)], [np.cos(theta)]]))
    #     print("dot prod = ", turn_delta*d.T@np.array([[-np.sin(theta)], [np.cos(theta)]]))
    #     print("angle = ", theta)
    #     print("x = ", x)
    #     print("y = ", y )
    #     print("dhdx = ", dhdx)
    #     print("g = ", g)
    bits.append([Lfh, Lgh, h])

    h1 = bits[0][2]
    h2 = bits[1][2]

    return min([h1,h2])

theta1 = np.arctan2(l / pi, l / 2)
theta2 = np.arctan2(l / pi, -l / 2)
theta3 = np.arctan2(-l / pi, -l / 2) + 2*pi
theta4 = np.arctan2(-l / pi, l / 2)
def inner_loop(thetas):
    r = l / pi
    rs = []
    for theta in thetas:
        if (theta4 < theta and theta < theta1):
            rs.append(l/2*np.cos(theta) + np.sqrt(r**2 - (l/2)**2*np.sin(theta)**2))
        elif (theta4+2*pi < theta and theta < theta1+2*pi):
            rs.append(l/2*np.cos(theta) + np.sqrt(r**2 - (l/2)**2*np.sin(theta)**2))
        elif theta1 < theta and theta < theta2:
            rs.append(r/np.sin(theta))
        elif theta2 < theta and theta < theta3:
            rs.append(-l/2*np.cos(theta) + np.sqrt(r**2 - (l/2)**2*np.sin(theta)**2))
        elif theta3 < theta and theta < theta4+2*pi:
            rs.append(-r/np.sin(theta))
        else:
            rs.append(0)
    return rs

o_theta1 = np.arctan2(l/pi+ w, l / 2)
o_theta2 = np.arctan2(l/pi + w, -l / 2)
o_theta3 = np.arctan2(-l/pi - w, -l / 2) + 2*pi
o_theta4 = np.arctan2(-l/pi - w, l / 2)
def outer_loop(thetas):
    r = l/pi + w
    rs = []
    for theta in thetas:
        if (o_theta4 < theta and theta < o_theta1):
            rs.append(l/2*np.cos(theta) + np.sqrt(r**2 - (l/2)**2*np.sin(theta)**2))
        elif (o_theta4+2*pi < theta and theta < o_theta1+2*pi):
            rs.append(l/2*np.cos(theta) + np.sqrt(r**2 - (l/2)**2*np.sin(theta)**2))
        elif o_theta1 < theta and theta < o_theta2:
            rs.append(r/np.sin(theta))
        elif o_theta2 < theta and theta < o_theta3:
            rs.append(-l/2*np.cos(theta) + np.sqrt(r**2 - (l/2)**2*np.sin(theta)**2))
        elif o_theta3 < theta and theta < o_theta4+2*pi:
            rs.append(-r/np.sin(theta))
        else:
            rs.append(0)
    return rs

wm = w/2
m_theta1 = np.arctan2(l/pi+ wm, l / 2)
m_theta2 = np.arctan2(l/pi + wm, -l / 2)
m_theta3 = np.arctan2(-l/pi - wm, -l / 2) + 2*pi
m_theta4 = np.arctan2(-l/pi - wm, l / 2)
def mid_loop(thetas):
    r = l/pi + wm
    rs = []
    for theta in thetas:
        if (m_theta4 < theta and theta < m_theta1):
            rs.append(l/2*np.cos(theta) + np.sqrt(r**2 - (l/2)**2*np.sin(theta)**2))
        elif (m_theta4+2*pi < theta and theta < m_theta1+2*pi):
            rs.append(l/2*np.cos(theta) + np.sqrt(r**2 - (l/2)**2*np.sin(theta)**2))
        elif m_theta1 < theta and theta < m_theta2:
            rs.append(r/np.sin(theta))
        elif m_theta2 < theta and theta < m_theta3:
            rs.append(-l/2*np.cos(theta) + np.sqrt(r**2 - (l/2)**2*np.sin(theta)**2))
        elif m_theta3 < theta and theta < m_theta4+2*pi:
            rs.append(-r/np.sin(theta))
        else:
            rs.append(0)
    return rs


def polar_to_xy(theta, r):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x,y

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
thetas = np.linspace(0,2*pi, 1000)
rs = inner_loop(thetas)
# plt.plot(thetas, rs)
xs, ys = polar_to_xy(thetas, rs)
ax.plot(xs, ys, 'k')
rs = outer_loop(thetas)
xs, ys = polar_to_xy(thetas, rs)
ax.plot(xs,ys, 'k')
rms = mid_loop(thetas)
xms, yms = polar_to_xy(thetas, rms)
ax.plot(xms,yms, 'y--')

end = 90
for i in range(54): 
    xs = np.load('plots/' +prepend + 'x_traj_'+str(i)+".np.npy")
    ax.plot(xs[:end,0], xs[:end,1], 'y')
for i in range(54): 
    xs = np.load('plots/' +prepend + 'x_traj_'+str(i)+".np.npy")
    ax.plot(xs[0,0], xs[0,1], 'b^')

plt.show()


saveFigure(fig, prepend + "full_track.pdf", '.')


minh = 100

dt = 1.0/60 
fig = plt.figure()
for i in range(54):
    hs = []
    ts = []
    xs = np.load('plots/'+prepend+'x_traj_'+str(i)+".np.npy")
    for j, state in enumerate(xs):
        hs.append(getCBFValue(state))
        ts.append(dt*j)

    plt.plot(ts, hs, color = [0.7,0.7,1])
    if minh > min(hs): 
        minh = min(hs)
        minh_idx = i 

i = minh_idx
print(i)
hs = []
ts = []
xs = np.load('plots/'+prepend+'x_traj_'+str(i)+".np.npy")
for j, state in enumerate(xs):
    if j == 0: 
        print(state)
    hs.append(getCBFValue(state))
    ts.append(dt*j)
    plt.plot(ts, hs, 'b')

plt.title("CBF Values $h(\mathbf{x})$ for several ICs. Min h = " + str(minh))
plt.xlabel("time [sec]")
plt.ylabel("$h$")
plt.ylim((-1, 5.5))
plt.hlines(0,ts[0], ts[-1], linestyle='--', color='r', linewidth=2)
plt.show()

saveFigure(fig, prepend + "hvals_fulltrack.pdf", '.')
