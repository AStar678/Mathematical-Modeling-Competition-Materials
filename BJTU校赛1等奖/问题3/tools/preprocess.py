import numpy as np
import math
from numpy import pi

#偏转角修改
angleA = 0
angleB = 2 * pi * (120 / 180)
angleC = 2 * pi * (230 / 180)


def getTheta(ka,kb,kc):
    kapp = (-(ka * np.cos(angleA) + kb * np.cos(angleB) + kc * np.cos(angleC)),-(ka * np.sin(angleA) + kb * np.sin(angleB) + kc * np.sin(angleC)))

    b = kapp[1]
    a = kapp[0]
    
    theta_radians = math.atan2(b, a)

    if theta_radians < 0:
        theta_radians = 2 * pi + theta_radians
    
    return theta_radians

def getSumk(ka,kb,kc):
    kapp = (-(ka * np.cos(angleA) + kb * np.cos(angleB) + kc * np.cos(angleC)),-(ka * np.sin(angleA) + kb * np.sin(angleB) + kc * np.sin(angleC)))
    sumk = math.sqrt(kapp[0]**2 + kapp[1]**2)
    return (2 / 3) * sumk