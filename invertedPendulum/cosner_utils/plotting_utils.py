import matplotlib.pyplot as plt
from cosner_utils.utils import *
import numpy as np
pi = np.pi


def inner_loop(thetas, l_fulltrack):
    l = l_fulltrack / 4
    theta1 = np.arctan2(l / pi, l / 2)
    theta2 = np.arctan2(l / pi, -l / 2)
    theta3 = np.arctan2(-l / pi, -l / 2) + 2 * pi
    theta4 = np.arctan2(-l / pi, l / 2)
    r = l / pi
    rs = []
    for theta in thetas:
        if (theta4 < theta and theta < theta1):
            rs.append(l / 2 * np.cos(theta) + np.sqrt(r ** 2 - (l / 2) ** 2 * np.sin(theta) ** 2))
        elif (theta4 + 2 * pi < theta and theta < theta1 + 2 * pi):
            rs.append(l / 2 * np.cos(theta) + np.sqrt(r ** 2 - (l / 2) ** 2 * np.sin(theta) ** 2))
        elif theta1 < theta and theta < theta2:
            rs.append(r / np.sin(theta))
        elif theta2 < theta and theta < theta3:
            rs.append(-l / 2 * np.cos(theta) + np.sqrt(r ** 2 - (l / 2) ** 2 * np.sin(theta) ** 2))
        elif theta3 < theta and theta < theta4 + 2 * pi:
            rs.append(-r / np.sin(theta))
        else:
            rs.append(0)
    return rs

def outer_loop(thetas, l_fulltrack, w):
    l = l_fulltrack/4
    o_theta1 = np.arctan2(l / pi + w, l / 2)
    o_theta2 = np.arctan2(l / pi + w, -l / 2)
    o_theta3 = np.arctan2(-l / pi - w, -l / 2) + 2 * pi
    o_theta4 = np.arctan2(-l / pi - w, l / 2)
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


def polar_to_xy(theta, r):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x,y


def plotTrack(car_x, car_y, l_fulltrack,w, data_output_path):
    """
    Plots and saves track image
    :param car_x: x position of the car as a (n,1) numpy array
    :param car_y: y position of the car as a (n,1) numpy array
    :param l: length of the track (ex 400m)
    :param w: width of the track (ex 6m 6 lane track)
    :param data_output_path: path to save data location
    :return: nothing
    """
    l = l_fulltrack/4
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



    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    thetas = np.linspace(0,2*pi, 1000)
    rs = inner_loop(thetas, l_fulltrack)
    # plt.plot(thetas, rs)
    xs, ys = polar_to_xy(thetas, rs)
    ax.plot(xs, ys, 'k')
    rs = outer_loop(thetas, l_fulltrack, w)
    xs, ys = polar_to_xy(thetas, rs)
    ax.plot(xs,ys, 'k')
    rms = mid_loop(thetas)
    xms, yms = polar_to_xy(thetas, rms)
    ax.plot(xms,yms, 'y--')
    ax.plot(car_x, car_y, 'b')
    plt.show()

    saveFigure(fig, "track.pdf", data_output_path)
