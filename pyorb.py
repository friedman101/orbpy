from numpy import *
from math import *

def orb2eci(SMA, ecc, inc, RAAN, arg_per, true_anom, mu):
	# from http://www.mathworks.com/matlabcentral/fileexchange/35455-convert-keplerian-orbital-elements-to-a-state-vector
	e = ecc
	i = inc
	O = RAAN
	o = arg_per
	nu = true_anom
	p = SMA*(1-e**2)

	#mu = 398600.4418; 

	rPQW = matrix([p*cos(nu)/(1 +e*cos(nu)),p*sin(nu)/(1+e*cos(nu)),0])
	vPQW = matrix([-sqrt(mu/p)*sin(nu),sqrt(mu/p)*(e+cos(nu)),0])

	PQW2IJK = zeros((3,3));
	cO = cos(O); sO = sin(O); co = cos(o); so = sin(o); ci = cos(i); si = sin(i)
	PQW2IJK[0,0] = cO*co-sO*so*ci
	PQW2IJK[0,1] = -cO*so-sO*co*ci
	PQW2IJK[0,2] = sO*si
	PQW2IJK[1,0] = sO*co+cO*so*ci
	PQW2IJK[1,1] = -sO*so+cO*co*ci
	PQW2IJK[1,2] = -cO*si
	PQW2IJK[2,0] = so*si
	PQW2IJK[2,1] = co*si
	PQW2IJK[2,2] = ci

	r = PQW2IJK*transpose(rPQW)
	v = PQW2IJK*transpose(vPQW)

	pos = squeeze(asarray(r))
	vel = squeeze(asarray(v))
	return (pos, vel)

def eci2orb(pos, vel, mu):
	# from http://www.mathworks.com/matlabcentral/fileexchange/35455-convert-keplerian-orbital-elements-to-a-state-vector
	r = pos
	v = vel
	# Specific angular momentum
	h = cross(r,v)
	n = cross([0,0,1],h)
	nMag = sqrt(dot(n,n))
	vMag = sqrt(dot(v,v))
	rMag = sqrt(dot(r,r))
	hMag = sqrt(dot(h,h))
	e = (1.0/mu)*((vMag**2 - mu/rMag)*r - dot(r,v)*v)
	eMag = sqrt(dot(e,e))
	zeta = (vMag**2)/2 - mu/rMag

	a = -mu/(2*zeta)
	p = a*(1-eMag**2)
	i = acos(h[2]/hMag);

	if n[0] != 0:
		O = acos(n[0]/nMag)
	else:
		O = 0

	if dot(n,e) != 0:
		o = acos(dot(n,e)/(nMag*eMag))
	else:
		o = 0

	if dot(e,r) != 0:
		nu = acos(dot(e,r)/(eMag*rMag))
	else:
		nu = 0;

	if e[0] != 0:
		lonPer = acos(e[0]/eMag)
	else:
		lonPer = 0

	if dot(n,r) != 0:
		argLat = acos(dot(n,r)/(nMag*rMag))
	else:
		argLat = 0

	truLon = acos(r[0]/rMag);

	# quadrant check
	if n[1] < 0:
		O = 2*pi-O
	if e[2] < 0:
		o = 2*pi-o 
	if dot(r,v) < 0:
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
	return residual

def anom_residual_by_true(true_anom,ecc):
	x = ((ecc*(cos(true_anom)+ecc)*sin(true_anom))/(ecc*cos(true_anom)+1)**2-sin(true_anom)/(ecc*cos(true_anom)+1))/sqrt(1-(cos(true_anom)+ecc)**2/(ecc*cos(true_anom)+1)**2)-(ecc*((ecc*(cos(true_anom)+ecc)*sin(true_anom))/(ecc*cos(true_anom)+1)**2-sin(true_anom)/(ecc*cos(true_anom)+1))*cos(acos((cos(true_anom)+ecc)/(ecc*cos(true_anom)+1))))/sqrt(1-(cos(true_anom)+ecc)**2/(ecc*cos(true_anom)+1)**2)
	return(x)

def true2mean_anom(true_anom, ecc):
	# from http://www.braeunig.us/space/orbmech.htm

	E = atan2(sqrt(1-ecc**2)*sin(true_anom),ecc+cos(true_anom))
	mean_anom = E - ecc*sin(E)
	if mean_anom < 0:
		mean_anom = 2*pi + mean_anom 
	return(mean_anom)

def mean2true_anom(mean_anom, ecc):
	# from http://www.braeunig.us/space/orbmech.htm

	max_iter = 5;
	tol = 1e-5;
	dx = 1e-9;

	if mean_anom < 0:
		mean_anom = 2*pi + true_anom

	if abs(mean_anom) < tol:
		return mean_anom

	true_anom = mean_anom
	# have to newton iterate as function isn't invertible
	for i in range(0,max_iter):
		f = anom_residual(true_anom, mean_anom, ecc)
		fp = anom_residual(true_anom+dx, mean_anom, ecc)
		fm = anom_residual(true_anom-dx, mean_anom, ecc)
		dfdx = (fp-fm)/(2*dx)
		true_anom = true_anom - f/dfdx
		if (abs(f) < tol):
			break

	if i==max_iter-1:
		print('badness')

	if true_anom < 0:
		true_anom = 2*pi + true_anom

	return (true_anom)

def twobody_prop(pos, vel, t, mu):
	# from http://www.braeunig.us/space/orbmech.htm

	(ecc, inc, RAAN, arg_per, true_anom, SMA) = eci2orb(pos, vel, mu)
	mean_anom0 = true2mean_anom(true_anom, ecc)
	n = sqrt(mu/SMA**3);
	mean_anom = mean_anom0 + n*t;
	true_anom = mean2true_anom(mean_anom, ecc)
	(pos, vel) = orb2eci(SMA, ecc, inc, RAAN, arg_per, true_anom, mu)
	return (pos, vel)