import numpy as np
import math
from math import pi

def eci2ecef(UTC, xp, yp, DUT1, DAT):
	# from http://aa.usno.navy.mil/publications/docs/Circular_179.pdf
	UT1 = UTC + DUT1
	TAI = UTC+DAT
	TT=TAI+32.184
	TBD=TT

	DU = unixtime2JD(UT1) - 2451545.0
	theta = 0.7790572732640 + 1.00273781191135448*DU
	T = (unixtime2JD(TBD) - 2451545.0)/36525
	GMST =  86400*theta+ (0.014506 + 4612.156534*T+ 1.3915817*T**2-0.00000044*T**3-0.000029956*T**4-0.0000000368*T**5)/15

	eps = 84381.406-46.836769*T-0.0001831*T**2+ 0.00200340*T**3-0.000000576*T**4-0.0000000434*T**5

def R1(x):
	y = np.array([ [1, 0, 0], [0, math.cos(x), math.sin(x)], [0, -math.sin(x), math.cos(x)] ])
	return(y)

def R2(x):
	y = np.array([ [math.cos(x), 0, -math.sin(x)], [0, 1, 0], [math.sin(x), 0, math.cos(x)] ])
	return(y)

def R3(x):
	y = np.array([ [math.cos(x), math.sin(x), 0], [-math.sin(x), math.cos(x), 0], [0, 0, 1] ])

def unixtime2JD(UT1):
	return ( unix_time / 86400.0 ) + 2440587.5

def cross_mat(x):
	if np.size(x)!=3:
		print("passed cross_mat non 3 vector")
	y = np.array([ [0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0] ])
	return(y)

def orb2eci(SMA, ecc, inc, RAAN, arg_per, true_anom, mu):
	# from http://www.mathworks.com/matlabcentral/fileexchange/35455-convert-keplerian-orbital-elements-to-a-state-vector
	e = ecc
	i = inc
	O = RAAN
	o = arg_per
	nu = true_anom
	p = SMA*(1-e**2) 

	rPQW = np.array([[p*math.cos(nu)/(1 +e*math.cos(nu)),p*math.sin(nu)/(1+e*math.cos(nu)),0]])
	vPQW = np.array([[-math.sqrt(mu/p)*math.sin(nu),math.sqrt(mu/p)*(e+math.cos(nu)),0]])

	PQW2IJK = np.zeros((3,3));
	cO = math.cos(O); sO = math.sin(O); co = math.cos(o); so = math.sin(o); ci = math.cos(i); si = math.sin(i)
	PQW2IJK[0,0] = cO*co-sO*so*ci
	PQW2IJK[0,1] = -cO*so-sO*co*ci
	PQW2IJK[0,2] = sO*si
	PQW2IJK[1,0] = sO*co+cO*so*ci
	PQW2IJK[1,1] = -sO*so+cO*co*ci
	PQW2IJK[1,2] = -cO*si
	PQW2IJK[2,0] = so*si
	PQW2IJK[2,1] = co*si
	PQW2IJK[2,2] = ci

	r = PQW2IJK.dot(rPQW.transpose())
	v = PQW2IJK.dot(vPQW.transpose())

	pos = np.squeeze(r)
	vel = np.squeeze(v)

	return(pos, vel)

def eci2orb(pos, vel, mu):
	# from http://www.mathworks.com/matlabcentral/fileexchange/35455-convert-keplerian-orbital-elements-to-a-state-vector
	r = pos
	v = vel
	# Specific angular momentum
	h = np.cross(r,v)
	n = np.cross([0,0,1],h)
	nMag = math.sqrt(np.dot(n,n))
	vMag = math.sqrt(np.dot(v,v))
	rMag = math.sqrt(np.dot(r,r))
	hMag = math.sqrt(np.dot(h,h))
	e = (1.0/mu)*((vMag**2 - mu/rMag)*r - np.dot(r,v)*v)
	eMag = math.sqrt(np.dot(e,e))
	zeta = (vMag**2)/2 - mu/rMag

	a = -mu/(2*zeta)
	p = a*(1-eMag**2)
	i = math.acos(h[2]/hMag);

	if n[0] != 0:
		O = math.acos(n[0]/nMag)
	else:
		O = 0

	if np.dot(n,e) != 0:
		o = math.acos(np.dot(n,e)/(nMag*eMag))
	else:
		o = 0

	if np.dot(e,r) != 0:
		nu = math.acos(np.dot(e,r)/(eMag*rMag))
	else:
		nu = 0;

	if e[0] != 0:
		lonPer = math.acos(e[0]/eMag)
	else:
		lonPer = 0

	if np.dot(n,r) != 0:
		argLat = math.acos(np.dot(n,r)/(nMag*rMag))
	else:
		argLat = 0

	truLon = math.acos(r[0]/rMag);

	# quadrant check
	if n[1] < 0:
		O = 2*pi-O
	if e[2] < 0:
		o = 2*pi-o 
	if np.dot(r,v) < 0:
		nu = 2*pi-nu
	if e[1] < 0:
		lonPer = 2*pi-lonPer
	if r[2]<0:
		argLat = 2*pi-argLat
	if r[1]<0:
		truLon = 2*pi-truLon

	ecc = eMag;
	inc = i;
	RAAN = O;
	arg_per = o;
	true_anom = nu;
	SMA = p/(1-eMag**2);
	return(ecc, inc, RAAN, arg_per, true_anom, SMA)

def anom_residual(true_anom,mean_anom,ecc):
	residual = mean_anom - true2mean_anom(true_anom, ecc)
	return(residual)

def true2mean_anom(true_anom, ecc):
	# from http://www.braeunig.us/space/orbmech.htm

	E = math.atan2(math.sqrt(1-ecc**2)*math.sin(true_anom),ecc+math.cos(true_anom))
	mean_anom = E - ecc*math.sin(E)
	if mean_anom < 0:
		mean_anom = 2*pi + mean_anom 
	return(mean_anom)

def mean2true_anom(mean_anom, ecc):
	# from http://www.braeunig.us/space/orbmech.htm

	max_iter = 5;
	tol = 1e-5;
	dx = 1e-9;

	if mean_anom < 0:
		mean_anom = 2*pi + mean_anom

	if abs(mean_anom) < tol:
		return mean_anom

	true_anom = mean_anom
	# have to newton iterate as function isn"t invertible
	for i in range(0,max_iter):
		f = anom_residual(true_anom, mean_anom, ecc)
		fp = anom_residual(true_anom+dx, mean_anom, ecc)
		fm = anom_residual(true_anom-dx, mean_anom, ecc)
		dfdx = (fp-fm)/(2*dx)
		true_anom = true_anom - f/dfdx
		if (abs(f) < tol):
			break

	if i==max_iter-1:
		print("max newton iterations exceeded")

	if true_anom < 0:
		true_anom = 2*pi + true_anom

	return(true_anom)

def twobody_prop(pos, vel, t, mu):
	(ecc, inc, RAAN, arg_per, true_anom, SMA) = eci2orb(pos, vel, mu)
	mean_anom0 = true2mean_anom(true_anom, ecc)
	n = math.sqrt(mu/SMA**3);
	mean_anom = mean_anom0 + n*t;
	true_anom = mean2true_anom(mean_anom, ecc)
	(pos, vel) = orb2eci(SMA, ecc, inc, RAAN, arg_per, true_anom, mu)
	return(pos, vel)