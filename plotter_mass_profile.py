import matplotlib.pyplot as plt
import numpy as np
import math
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

xpoints = []
ypoints = []
ypoints_ = []
# tmax = 9.42
tmax = 10 # for fast mass change test only
for t in range(int(tmax*100)):
    # if t >= 200 and t < 400:
    #     ypoints.append(1.3)
    # elif t >= 600:
    #     ypoints.append(0.65)
    # else:
    #     ypoints.append(1)
    # ypoints.append(9.34-1.5*math.sin(t/100)**2+2*math.cos(8*t/100)**4)
    # ypoints_.append(7.34-1.5*math.sin(t/100)**2+2*math.cos(8*t/100)**4)
    ### the following two lines plot the mass profile for 'drama_change_aggres'
    ypoints.append(5.34-1.5*math.sin(t/100)**2+2*math.cos(8*t/100)**4)
    ypoints_.append(3.34-1.5*math.sin(t/100)**2+2*math.cos(8*t/100)**4)

    ### the following two lines plot the mass profile for 'undrama_change'
    # ypoints.append(5.34-0.025*(t/100)**2+2*math.cos(2*t/100)-0.25*(t/100))
    # ypoints_.append(3.34-0.025*(t/100)**2+2*math.cos(2*t/100)-0.25*(t/100))
    xpoints.append(t/100)
fig, ax = plt.subplots()
# plt.title('Mass profile', fontdict=font)
# plt.xlabel('time (s)', fontdict=font)
# plt.ylabel('mass  (kg)', fontdict=font)
plt.title('Mass profile')
plt.xlabel('t [sec]')
plt.ylabel('m  [kg]')

plt.xlim(0,10)
plt.plot(xpoints, ypoints,'-',color="blue")
plt.plot(xpoints, ypoints_,'-',color="blue")
plt.fill_between(xpoints,ypoints,ypoints_)
plt.show()

# test code