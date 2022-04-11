import matplotlib.pyplot as plt
import numpy as np
import os
pi = np.pi



def rOfClosestPointInLine(theta, l,w):
    """
    Get radius of closest point in line
    :param theta:
    :param l:
    :param w:
    :return:
    """
    l /=4
    wm = w / 2
    m_theta1 = np.arctan2(l / pi + wm, l / 2)
    m_theta2 = np.arctan2(l / pi + wm, -l / 2)
    m_theta3 = np.arctan2(-l / pi - wm, -l / 2) + 2 * pi
    m_theta4 = np.arctan2(-l / pi - wm, l / 2)
    r = l / pi + wm

    def getCircleDir(rad, thet):
        """

        :param rad:
        :param thet:
        :return: perpendicular direction to track at point (rad, thet), points outwards
        """
        x = rad*np.cos(thet)
        y = rad*np.sin(thet)
        if x > 0:
            circ_theta = np.arctan2(y, x-l/2)
        else:
            circ_theta = np.arctan2(y, x + l/2)
        return np.array([ np.cos(circ_theta),np.sin(circ_theta)])

    if m_theta4 < theta < m_theta1:
        rs = (l / 2 * np.cos(theta) + np.sqrt(r ** 2 - (l / 2) ** 2 * np.sin(theta) ** 2))
        dir = getCircleDir(rs,theta)
    elif m_theta4 + 2 * pi < theta < m_theta1 + 2 * pi:
        rs = (l / 2 * np.cos(theta) + np.sqrt(r ** 2 - (l / 2) ** 2 * np.sin(theta) ** 2))
        dir = getCircleDir(rs,theta)
    elif m_theta1 < theta < m_theta2:
        rs = (r / np.sin(theta))
        dir = np.array([[0],[1]])
    elif m_theta2 < theta < m_theta3:
        rs = (-l / 2 * np.cos(theta) + np.sqrt(r ** 2 - (l / 2) ** 2 * np.sin(theta) ** 2))
        dir = getCircleDir(rs,theta)
    elif m_theta3 < theta < m_theta4 + 2 * pi:
        rs = (-r / np.sin(theta))
        dir = np.array([[0],[-1]])

    return rs, dir

def saveFigure(fig, name, path):
    os.system("mkdir "+ path + "/plots")
    print("saving to " + path+"/plots/"+name)
    fig.savefig(path+"/plots/"+name)

def plotTrack(car_x, car_y, l,w, data_output_path, save_flag, idx, is_pdf):
    """
    Plots and saves track image
    :param car_x: x position of the car as a (n,1) numpy array
    :param car_y: y position of the car as a (n,1) numpy array
    :param l: length of the track (ex 400m)
    :param w: width of the track (ex 6m 6 lane track)
    :param data_output_path: path to save data location
    :return: nothing
    """
    l = l/4
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
    ax.plot(car_x, car_y, 'b')
    plt.show()

    if save_flag:
        if is_pdf: 
            saveFigure(fig, "track_" + str(idx) + ".pdf", data_output_path)
        else:
            saveFigure(fig, "track_" + str(idx), data_output_path)



