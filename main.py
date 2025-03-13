# AEM 525 HW7 Holden McNerney

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npla

def problem_1():
    '''
    Problem 1
    '''
    p1_m = 3
    s1_m = 6
    s2_m = 2

    r_p1 = np.array([[0, 0]])
    r_s1 = np.array([[-6, 3]])
    r_s2 = np.array([[-4, -1]])

    x_cm = (r_s1[0,0]*s1_m + r_s2[0,0]*s2_m)/(s1_m + s2_m)
    y_cm = (r_s1[0,1]*s1_m + r_s2[0,1]*s2_m)/(s1_m + s2_m)

    G = 6.67*10**-11

    F = -G*p1_m*(s1_m*r_s1*npla.norm(r_s1)**-3 + s2_m*r_s2*npla.norm(r_s2)**-3)
    F_unit = F/npla.norm(F)
    R_cg = np.sqrt(G*p1_m*(s1_m + s2_m)/npla.norm(F))

    x_cg = -R_cg*F_unit[0,0]
    y_cg = -R_cg*F_unit[0,1]

    M = np.cross(np.array([[x_cm - x_cg, y_cm - y_cg]]), F)
    print(M)

    plt.plot([0, x_cg], [0, y_cg], 'm', label='Line of Action')
    plt.plot(r_p1[0,0], r_p1[0,1], 'bo', label="P'")
    plt.plot(r_s1[0,0], r_s1[0,1], 'go', label="S1")
    plt.plot(r_s2[0,0], r_s2[0,1], 'ro', label="S2")
    plt.quiver(0, 0, F_unit[0,0], F_unit[0,1])
    plt.plot(x_cm, y_cm, 'yo', label="cm")
    plt.plot(x_cg, y_cg, 'co', label="cg")

    plt.legend()
    # plt.show()

    return 0

def problem_2():
    '''
    Testing approximation divergence
    '''
    G = 1
    M = 1
    m = 1
    R = 1
    F_exact = lambda r: -G*M*m*(R**2 + r**2)**(-3/2)
    F_approx = lambda r: -G*M*m/R**2 - 3/2*G*M/R**4*(-3*m*r**2) - 3/R**4*G*M*m*r**2
    i = np.arange(0, 1, 0.05)
    plt.plot(i, F_exact(i), label='Exact')
    plt.plot(i, F_approx(i), label='Approx')
    plt.legend()

    plt.show()
    return 0

def problem_3():
    '''
    Problem 3
    '''
    R = 6498
    L = 50
    F_part = 1/R**2*(1 - 3/2*10/48*L**2/R**2 + L**2/(4*R**2))
    R_cg = np.sqrt(1/F_part)
    print(F_part)
    print(R_cg)

    return 0

def problem_4():
    """
    Solution to Problem 4
    """
    dt = 0.001
    I = 300
    J = 100
    k = 2*(J - I)/I
    T = 10
    omega_0_vec = np.array([[1, -1, 2]])
    ep_0_vec = np.array([[0, 0, 0, 1]])

    # Initialize History Arrays
    t_vec = np.arange(0, 15, dt)
    omega_hist = np.zeros((len(t_vec),3))
    ep_hist = np.zeros((len(t_vec),4))
    C_hist = np.zeros((len(t_vec),3))
    precess_hist = np.zeros((len(t_vec)))
    nutat_hist = np.zeros((len(t_vec)))

    omega = omega_0_vec.astype(float)
    ep = ep_0_vec.astype(float)

    for i, t in enumerate(t_vec):
        if i == 0:
            omega_hist[0, :] = omega
            ep_hist[0, :] = ep
            C_hist[0, :] = convert_ep_to_dcm(ep)
            nutat_hist[0] = 0
            precess_hist[0] = -45*np.pi/180                                  
        else:
            ep_dot = calc_ep_dot(ep, omega)
            delta_ep = ep_dot.T*dt
            ep += delta_ep
            ep = ep/npla.norm(ep)
            ep_hist[i, :] = ep

            omega_dot = calc_omega_dot(T, I, k, omega)
            delta_omega = omega_dot.T*dt
            omega += delta_omega
            omega_hist[i, :] = omega

            C_hist[i, :] = convert_ep_to_dcm(ep)
            nutat_hist[i] = np.arccos(C_hist[i, 2])
            if C_hist[i, 0] >= 0 and C_hist[i, 1] >= 0:
                precess_hist[i] = -np.pi/2 + np.arccos(-C_hist[i, 1]/np.sin(nutat_hist[i]))
                pass
            elif C_hist[i, 0] <= 0 and C_hist[i, 1] >= 0:
                precess_hist[i] = np.pi/2 - np.arcsin(C_hist[i, 0]/np.sin(nutat_hist[i]))
                pass
            elif C_hist[i, 0] <= 0 and C_hist[i, 1] <= 0:
                precess_hist[i] = -np.pi/2 -np.arccos(-C_hist[i, 1]/np.sin(nutat_hist[i]))
                pass
            else:
                precess_hist[i] = -np.pi/2 + np.arccos(-C_hist[i, 1]/np.sin(nutat_hist[i]))
                pass

            if t == 2.4:
                t_2_4_C13 = C_hist[i, 0]
                t_2_4_C23 = C_hist[i, 1]
                t_2_4_precess = precess_hist[i]
                t_2_4_nutat = nutat_hist[i]
            if t == 8.2:
                t_8_2_C13 = C_hist[i, 0]
                t_8_2_C23 = C_hist[i, 1]
                t_8_2_precess = precess_hist[i]
                t_8_2_nutat = nutat_hist[i]

    print('Using 3-1-3')
    print(f'''Precession Angle at t = 2.4: {t_2_4_precess*180/np.pi}''')
    print(f'''Precession Angle at t = 8.2: {t_8_2_precess*180/np.pi}''')

    print('Using 3-2-3')
    t_2_4_precess = -np.arccos(t_2_4_C13/np.sin(t_2_4_nutat))
    t_8_2_precess = -np.arccos(t_8_2_C13/np.sin(t_8_2_nutat))
    print(f'''Precession Angle at t = 2.4: {t_2_4_precess*180/np.pi}''')
    print(f'''Precession Angle at t = 8.2: {t_8_2_precess*180/np.pi}''')


    # PLOT OF C23 VS C13
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.plot(C_hist[:, 0], C_hist[:, 1], label=r'$C_{23}$ vs $C_{13}$')
    plt.plot(0, 0, 'ro', label=r'Starting Point, $t_0$')
    plt.quiver(0, 0, 0, 1, scale=5, label='Inertial Frame')
    plt.quiver(0, 0, 1, 0, scale=5)
    plt.text(0.35, -0.025, r'$\hat{n}_1$')
    plt.text(0.015, 0.335, r'$\hat{n}_2$')
    # Body Vectors are t = 2.4 seconds
    plt.plot([0, t_2_4_C13], [0, t_2_4_C23])
    plt.text(t_2_4_C13 + 0.05, t_2_4_C23, 't=2.4s')
    plt.quiver(t_2_4_C13, t_2_4_C23, -np.sin(t_2_4_precess), np.cos(t_2_4_precess), 
               scale=5, color='b', label=r"Body Frame, b'")
    plt.quiver(t_2_4_C13, t_2_4_C23, np.cos(t_2_4_precess), np.sin(t_2_4_precess), 
               scale=5, color='b')
    plt.text(0.55, -0.145, r"$\hat{b}_2'$")
    plt.text(0.88, 0.21, r"$\hat{b}_1'$")
    plt.plot(t_2_4_C13, t_2_4_C23, 'go', label=r'$t_{2.4}$')
    # Body Vectors are t = 8.2 seconds
    plt.plot([0, t_8_2_C13], [0, t_8_2_C23])
    plt.text(t_8_2_C13 - 0.15, t_8_2_C23, 't=8.2s')
    plt.quiver(t_8_2_C13, t_8_2_C23, -np.sin(t_8_2_precess), np.cos(t_8_2_precess), 
               scale=5, color='m', label=r"Body Frame, b'")
    plt.quiver(t_8_2_C13, t_8_2_C23, np.cos(t_8_2_precess), np.sin(t_8_2_precess), 
               scale=5, color='m')
    plt.text(0.05, -0.35, r"$\hat{b}_1'$")
    plt.text(0.16, 0.05, r"$\hat{b}_2'$")
    plt.plot(t_8_2_C13, t_8_2_C23, 'go', label=r'$t_{8.2}$')
    plt.ylabel('C23')
    plt.xlabel('C13')
    plt.xlim([-0.6, 1.3])
    plt.ylim([-1.15, 0.6])
    plt.title(r'$C_{13}$ vs $C_{23}$ with Inertial and Body Frame Vectors')
    plt.legend()
    plt.show()

    return 0

def calc_ep_dot(e, omega):
    '''
    Calculates the rate of change of euler parameters
    '''
    e_dot = np.array([[1/2*(omega[0,0]*e[0,3] - omega[0,1]*e[0,2] + omega[0,2]*e[0,1])],
                      [1/2*(omega[0,0]*e[0,2] + omega[0,1]*e[0,3] - omega[0,2]*e[0,0])],
                      [1/2*(-omega[0,0]*e[0,1] + omega[0,1]*e[0,0] + omega[0,2]*e[0,3])],
                      [-1/2*(omega[0,0]*e[0,0] + omega[0,1]*e[0,1] + omega[0,2]*e[0,2])]])

    return e_dot

def calc_omega_dot(T, I, k, omega):
    '''
    Calculates the rate of change of angular rates
    '''
    omega_dot = np.array([[T/I - k*omega[0,1]],
                          [k*omega[0,0]],
                          [0]])

    return omega_dot

def convert_ep_to_dcm(e):
    '''
    Converts Euler Parameters to DCM values
    '''
    C13 = 2*(e[0,2]*e[0,0] + e[0,1]*e[0,3])
    C23 = 2*(e[0,1]*e[0,2] - e[0,0]*e[0,3])
    C33 = 1 - 2*e[0,0]**2 - 2*e[0,1]**2
    C = np.array([C13, C23, C33])

    return C

def main():

    # problem_1()
    # problem_2()
    # problem_3()
    problem_4()

    return 0

if __name__=='__main__':
    main()