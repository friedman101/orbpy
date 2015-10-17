from numpy import *

def orb2eci(SMA, ecc, inc, RAAN, arg_per, true_anom):
	e = ecc
	i = inc
	O = RAAN
	o = arg_per
	nu = true_anom
	p = SMA*(1-e**2)

	mu = 398600.4418; 

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

def eci2orb(pos, vel):
	r = pos
	v = vel
	mu = 398600.4418
	# Specific angular momentum
	h = cross(r,v)
	n = cross([0,0,1],h)
	nMag = sqrt(dot(n,n))
	vMag = sqrt(dot(v,v))
	rMag = sqrt(dot(r,r))
	hMag = sqrt(dot(h,h))
	e = (1/mu)*((vMag**2 - mu/rMag)*r - dot(r,v)*v)
	eMag = sqrt(dot(e,e))
	zeta = (vMag**2)/2 - mu/rMag
	# idx = eMag ~= 1
	a = -mu/(2*zeta)
	p = a*(1-eMag**2)
	i = acos(h[2]/hMag);
	O = acos(n[0]/nMag);
	o = acos(dot(n,e)/(nMag*eMag));
	nu = acos(dot(e,r)/(eMag*rMag));
	lonPer = acos(e[0]/eMag);
	argLat = acos(dot(n,r)/(nMag*rMag));
	truLon = acos(r[0]/rMag);

	ecc = eMag;
	inc = i;
	RAAN = O;
	arg_per = o;
	true_anom = nu;
	SMA = p/(1-eMag**2);
	return(ecc, inc, RAAN, arg_per, true_anom, SMA)