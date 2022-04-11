import numpy as np
import scipy as sp
import ecos
pi = np.pi

"""
Race Track Controller
"""
l = 20  # length of the track (ex 400 m )
w = 2  # width of track (ex 6m)
scale = 2
Kp = 0.1 * scale
Kr = 0.5 * scale
FF = 1 * scale
Kdir = 0.2 * scale


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

    return u

"""
CBFQP filtered race track controller
"""
alpha = 1
def raceTrackBarrierBits(state):
    l = 20  # length of the track (ex 400 m )
    w = 2  # width of track (ex 6m)
    l /= 4
    x = state[0, 0]
    y = state[1, 0]
    theta = state[2, 0]

    dhdx = np.zeros((1, 3))
    r = l / pi
    if x >= l / 2:
        h = (r + w) ** 2 - ((x - l / 2) ** 2 + y ** 2)
        dhdx[0, 0] = -2 * (x - l / 2)
        dhdx[0, 1] = -2 * y
    elif x <= -l / 2:
        h = (r + w) ** 2 - ((x + l / 2) ** 2 + y ** 2)
        dhdx[0, 0] = -2 * (x + l / 2)
        dhdx[0, 1] = -2 * y
    elif y <= 0:
        h = y + (r + w)
        dhdx[0, 1] = 1
    elif y > 0:
        h = (r + w) - y
        dhdx[0, 1] = -1

    g = np.array([
        [np.cos(theta), 0],
        [np.sin(theta), 0],
        [0, 1]
    ])

    Lgh = dhdx @ g

    return Lgh, h


def raceTrackBarrierPartialBits(state):
    l = 20  # length of the track (ex 400 m )
    w = 2  # width of track (ex 6m)
    l /= 4
    x = state[0, 0]
    y = state[1, 0]
    theta = state[2, 0]

    dhdx = np.zeros((1, 3))
    r = l / pi
    dLghdx = np.zeros((1, 3, 2))
    if x >= l / 2:
        dhdx[0, 0] = -2 * (x - l / 2)
        dhdx[0, 1] = -2 * y
        dLghdx[0,0,0] = -2 * np.cos(theta)
        dLghdx[0,0,1] = -2 * np.sin(theta)
        dLghdx[0,0,2] = 2*(x-l/2)*np.sin(theta) - 2*y*np.cos(theta)
    elif x <= -l / 2:
        dhdx[0, 0] = -2 * (x + l / 2)
        dhdx[0, 1] = -2 * y
        dLghdx[0, 0, 0] = -2 * np.cos(theta)
        dLghdx[0, 0, 1] = -2 * np.sin(theta)
        dLghdx[0, 0, 2] = 2 * (x + l / 2) * np.sin(theta) - 2 * y * np.cos(theta)
    elif y <= 0:
        dhdx[0, 1] = 1
    elif y > 0:
        dhdx[0, 1] = -1

    return dhdx, dLghdx

def raceTrackCBFQP(state):
    knom = raceTrackController(state)
    Lgh, h = raceTrackBarrierBits(state)
    if Lgh@knom >= -alpha*h:
        return knom
    else:
        u = knom - Lgh.T/(Lgh@Lgh.T)*(Lgh@knom+alpha*h)
        return u


def raceTrackTROP(state):
    knom = raceTrackController(state)
    Lgh, h = raceTrackBarrierBits(state)

    epsilon = 0.05
    l = 20
    l /=4
    MRCBF_add = epsilon*2*(l/pi + l/2)
    MRCBF_mult = epsilon*2

    u_filtered  = K_CBF_SOCP(Lgh, h, knom, alpha, sigma=1, MRCBF_add=0.25, MRCBF_mult=0.2)

    return u_filtered

def K_CBF_SOCP(Lgh, h, u_des, alpha, sigma, MRCBF_add, MRCBF_mult):
    R = 1
    G = [[-1 / np.sqrt(2), 0, 0],
         [-1 / np.sqrt(2), 0, 0],
         [0, -1, 0],
         [0, 0, -R]]

    b = [1 / np.sqrt(2),
         -1 / np.sqrt(2),
         0,
         0]
    cones = [4]

    # for bit in barrier_bits:
    cones.append(3)
    #h = h
    Lfh = 0
    #Lgh = Lgh
    LghLgh = Lgh@Lgh.T
    G.append([0, -Lgh[0, 0], -Lgh[0, 1]])
    G.append([0, -MRCBF_mult, 0])
    G.append([0, 0, -MRCBF_mult])
    b.append((Lfh + alpha * h - (sigma * MRCBF_mult + MRCBF_add) - sigma * LghLgh).item())
    b.append(0)
    b.append(0)
    G = sp.sparse.csc_matrix(G)
    b = np.array(b)

    SOCP_dims = {
        'l': 0,  # linear cone size
        'q': cones,  # second order cone size
        'e': 0  # exponential cone sizes
    }

    cost = np.array([1.0, -u_des[0].item(), R ** 2 * -u_des[1].item()])

    ecos_solver_output = ecos.solve(cost, G, b, SOCP_dims, verbose=False)

    if ecos_solver_output['info']['exitFlag'] == 0 or ecos_solver_output['info']['exitFlag'] == 10:
        # ECOS Solver Successful
        return np.expand_dims(ecos_solver_output['x'][1:3], 1)
    else:
        # ECOS Solver Failed
        rospy.logwarn('SOCP failed')  # Filter falls back to zero input
        return np.array([[0, 0]]).T




