import numpy as np
from cosner_utils.plotting_utils import *
import scipy as sp
import ecos
pi = np.pi

TURN_CBF = True
turn_delta = 0.1


def rOfClosestPointInLine(theta, l,w, offset = 0 ):
    """
    Get radius of closest point in line
    :param theta:
    :param l:
    :param w:
    :return:
    """
    l /=4
    wm = w / 2
    theta = theta + offset
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
        dir = np.array([[-np.sin(-offset)],[np.cos(-offset)]])
    elif m_theta2 < theta < m_theta3:
        rs = (-l / 2 * np.cos(theta) + np.sqrt(r ** 2 - (l / 2) ** 2 * np.sin(theta) ** 2))
        dir = getCircleDir(rs,theta)
    elif m_theta3 < theta < m_theta4 + 2 * pi:
        rs = (-r / np.sin(theta))
        dir = np.array([[np.sin(-offset)],[-np.cos(-offset)]])

    return rs, dir



"""
Race Track Controller
"""
l = 20  # length of the track (ex 400 m )
l_fulltrack = l
w = 2  # width of track (ex 6m)
scale = 2
Kp = 0.2 * scale
Kr = 0.5 * scale # Kr = 0.5 * scale
FF = 1 * scale
Kdir = 0.2 * scale # Kdir = 0.2 * scale


def raceTrackController(x):
    m = 2
    u = np.zeros((m, 1))
    theta = np.arctan2(x[1], x[0])
    r = np.linalg.norm(x[0:2])
    if theta < 0:
        theta += 2 * np.pi

    r_midline, perp_midline = rOfClosestPointInLine(theta, l, w)

    dir = np.array([np.cos(x[2]), np.sin(x[2])])

    u[0, 0] = Kp * np.abs((r - r_midline)) + FF
    u[1, 0] = Kr * (r - r_midline) + Kdir * (dir.T @ perp_midline)

    return u, True

"""
CBFQP filtered race track controller
"""
alpha = 10

def raceTrackBarrierBits(state):

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

    return bits

def raceTrackBarrierPartialBits(state):
    l = l_fulltrack  # length of the track (ex 400 m )
    w = 2  # width of track (ex 6m)
    l /= 4
    x = state[0, 0]
    y = state[1, 0]
    theta = state[2, 0]

    dhdx = np.zeros((1, 3))
    r = l / pi
    dLghdx = np.zeros((2,3))
    if x >= l / 2:
        dhdx[0, 0] = -2 * (x - l / 2)
        dhdx[0, 1] = -2 * y
        dLghdx[0,0] = -2 * np.cos(theta)
        dLghdx[0,1] = -2 * np.sin(theta)
        dLghdx[0,2] = 2*(x-l/2)*np.sin(theta) - 2*y*np.cos(theta)
    elif x <= -l / 2:
        dhdx[0, 0] = -2 * (x + l / 2)
        dhdx[0, 1] = -2 * y
        dLghdx[0, 0] = -2 * np.cos(theta)
        dLghdx[0, 1] = -2 * np.sin(theta)
        dLghdx[0, 2] = 2 * (x + l / 2) * np.sin(theta) - 2 * y * np.cos(theta)
    elif y <= 0:
        dhdx[0, 1] = 1
    elif y > 0:
        dhdx[0, 1] = -15

    return dhdx, dLghdx

# Don't use this unless you need to run things at crazy speeds!
# Use trop with params set to 0
# def raceTrackCBFQP(state):
#     knom = raceTrackController(state)
#     Lgh, h = raceTrackBarrierBits(state)
#     if Lgh@knom >= -alpha*h:
#         return knom
#     else:
#         u = knom - Lgh.T/(Lgh@Lgh.T)*(Lgh@knom+alpha*h)
#         return u


def raceTrackCBFQP(state):
    knom, _ = raceTrackController(state)
    bits = raceTrackBarrierBits(state)
    u_filtered, flag= K_CBF_SOCP(bits, knom, alpha, sigma=0, MRCBF_add=0, MRCBF_mult=0)
    return u_filtered, flag


def raceTrackTROP(state):
    knom, _ = raceTrackController(state)
    bits = raceTrackBarrierBits(state)

    epsilon = 0.0#0.01
    l = l_fulltrack/4
    MRCBF_add = 0.01 #epsilon*2*(l/pi + l/2 + turn_delta)
    MRCBF_mult = 0.0001 #epsilon*(2 + turn_delta)
    sigma = 0.5

    u_filtered = K_CBF_SOCP(bits, knom, alpha, sigma, MRCBF_add, MRCBF_mult)

    return u_filtered

# def K_CBF_SOCP(Lgh, h, u_des, alpha, sigma, MRCBF_add, MRCBF_mult):
#     R = 1
#     G = [[-1 / np.sqrt(2), 0, 0],
#          [-1 / np.sqrt(2), 0, 0],
#          [0, -1, 0],
#          [0, 0, -R]]
#
#     b = [1 / np.sqrt(2),
#          -1 / np.sqrt(2),
#          0,
#          0]
#     cones = [4]
#
#     # for bit in barrier_bits:
#     cones.append(3)
#     #h = h
#     Lfh = 0
#     #Lgh = Lgh
#     LghLgh = Lgh@Lgh.T
#     G.append([0, -Lgh[0, 0], -Lgh[0, 1]])
#     G.append([0, -MRCBF_mult, 0])
#     G.append([0, 0, -MRCBF_mult])
#     b.append((Lfh + alpha * h - (sigma * MRCBF_mult + MRCBF_add) - sigma * LghLgh).item())
#     b.append(0)
#     b.append(0)
#     G = sp.sparse.csc_matrix(G)
#     b = np.array(b)
#
#     SOCP_dims = {
#         'l': 0,  # linear cone size
#         'q': cones,  # second order cone size
#         'e': 0  # exponential cone sizes
#     }
#
#     cost = np.array([1.0, -u_des[0].item(), R ** 2 * -u_des[1].item()])
#
#     ecos_solver_output = ecos.solve(cost, G, b, SOCP_dims, verbose=False)
#
#     if ecos_solver_output['info']['exitFlag'] == 0 or ecos_solver_output['info']['exitFlag'] == 10:
#         # ECOS Solver Successful
#         return np.expand_dims(ecos_solver_output['x'][1:3], 1)
#     else:
#         # ECOS Solver Failed
#         rospy.logwarn('SOCP failed')  # Filter falls back to zero input
#         return np.array([[0, 0]]).T



def K_CBF_SOCP(barrier_bits, u_des, alpha, sigma, MRCBF_add, MRCBF_mult):
    # Bits come in the form [Lfh, Lgh, h]. Lfh and h are scalars, Lgh is a 1xm numpy array
    R = 1
    m = len(u_des)

    G = []
    for i in range(2):
        line = [-1 / np.sqrt(2)]
        for j in range(m):
            line.append(0)
        G.append(line)
    cones = [2 + m ]

    for i in range(m):
        line = []
        for j in range(m+1):
            if i+1 == j:
                line.append(-1)
            else:
                line.append(0)
        G.append(line)
    b = [1 / np.sqrt(2),
         -1 / np.sqrt(2)
         ]
    for i in range(m):
        b.append(0)


    for bit in barrier_bits:
        Lfh = bit[0]
        Lgh = bit[1]
        h = bit[2]
        LghLgh = Lgh@Lgh.T

        line = [0]
        for i in range(m):
            line.append(-Lgh[0,i])
        G.append(line)

        for i in range(m):
            line = [0]
            for j in range(m):
                if i == j:
                    line.append(-MRCBF_mult)
                else:
                    line.append(0)
            G.append(line)

        b.append((Lfh + alpha * h - (sigma * MRCBF_mult + MRCBF_add) - sigma * LghLgh).item())
        for i in range(m):
            b.append(0)

        cones.append(m+1)

    G = sp.sparse.csc_matrix(G)
    b = np.array(b)

    SOCP_dims = {
        'l': 0,  # linear cone size
        'q': cones,  # second order cone size
        'e': 0  # exponential cone sizes
    }

    cost = [1.0]
    for i in range(m):
        cost.append(-u_des[i].item())

    cost = np.array(cost)

    ecos_solver_output = ecos.solve(cost, G, b, SOCP_dims, verbose=False)

    if ecos_solver_output['info']['exitFlag'] == 0 or ecos_solver_output['info']['exitFlag'] == 10:
        # ECOS Solver Successful
        return np.expand_dims(ecos_solver_output['x'][1:3], 1), True
    else:
        # ECOS Solver Failed
        print('SOCP failed')  # Filter falls back to zero input
        print(barrier_bits)
        u =  np.zeros((m,1))
        return u, False






