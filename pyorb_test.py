#!/usr/bin/python

from pyorb import *
from matplotlib.pyplot import *
from numpy import *

mu = 398600.4418
SMA = 7000;
ecc = 0.001;
inc = .1;
RAAN = .2;
arg_per = .3;
true_anom = .4;

(pos0,vel0) = orb2eci(SMA, ecc, inc, RAAN, arg_per, true_anom, mu)

t = arange(0,1*3600,30)
pos = zeros((size(t),3))
vel = zeros((size(t),3))

pos[0,] = pos0
vel[0,] = vel0

for i in range(1,size(t)):
	(pos_next, vel_next) = twobody_prop(pos0, vel0, t[i], mu)
	pos[i,] = pos_next
	vel[i,] = vel_next

plot(t,pos)
show()
