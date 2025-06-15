import numpy as np
import math
from numpy import pi

def getTheta(ka,kb,kc):
    kapp = (-(ka * np.cos(0) + kb * np.cos(2 * pi / 3) + kc * np.cos(4 * pi / 3)),-(ka * np.sin(0) + kb * np.sin(2 * pi / 3) + kc * np.sin(4 * pi / 3)))

    b = kapp[1]
    a = kapp[0]
    
    theta_radians = math.atan2(b, a)

    if theta_radians < 0:
        theta_radians = 2 * pi + theta_radians
    
    return theta_radians

def getSumk(ka,kb,kc):
    kapp = (-(ka * np.cos(0) + kb * np.cos(2 * pi / 3) + kc * np.cos(4 * pi / 3)),-(ka * np.sin(0) + kb * np.sin(2 * pi / 3) + kc * np.sin(4 * pi / 3)))
    sumk = math.sqrt(kapp[0]**2 + kapp[1]**2)
    return (2 / 3) * sumk