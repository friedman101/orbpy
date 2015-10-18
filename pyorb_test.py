#!/usr/bin/python

import pyorb as po
import matplotlib.pyplot as pp
import numpy as np

mu = 398600.4418
SMA = 7000;
ecc = 0.001;
inc = .1;
RAAN = .2;
arg_per = .3;
true_anom = .4;

(pos0,vel0) = po.orb2eci(SMA, ecc, inc, RAAN, arg_per, true_anom, mu)

t = np.arange(0,1*3600,30)
pos = np.zeros((np.size(t),3))
vel = np.zeros((np.size(t),3))

pos[0,] = pos0
vel[0,] = vel0

for i in range(1,np.size(t)):
	(pos_next, vel_next) = po.twobody_prop(pos0, vel0, t[i], mu)
	pos[i,] = pos_next
	vel[i,] = vel_next

pp.subplot(2,1,1)
pp.plot(t,pos)
pp.grid("on")
pp.ylabel("pos (km)")
pp.title("position in ECI")
pp.subplot(2,1,2)
pp.plot(t,vel)
pp.grid("on")
pp.ylabel("vel (km/s)")
pp.xlabel("time (s)")
pp.title("velocity in ECI")

x1 = np.array([-4, 5, 6]);
x2 = np.array([1, 2, 12]);
print("cross(x1, x2):")
print(np.cross(x1,x2))
print("cross_mat(x1)*x2")
print(po.cross_mat(x1).dot(x2))

pp.show()