# AEM 525 HW7 Holden McNerney

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npla

def problem1():
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

def problem3():
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

def main():

    # problem1()
    problem3()

    return 0

if __name__=='__main__':
    main()