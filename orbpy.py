import numpy as np
import math
from math import pi,sin,cos

def eci2ecef(UTC, xp, yp, DUT1, DAT, nutation_series):
        # from http://aa.usno.navy.mil/publications/docs/Circular_179.pdf
        UT1 = UTC + DUT1
        TAI = UTC + DAT
        TT = TAI + 32.184
        TBD = TT

	DU = unixtime2JD(UT1) - 2451545.0
        theta = 0.7790572732640 + 1.00273781191135448*DU
        T = (unixtime2JD(TBD) - 2451545.0)/36525
        GMST =  86400*theta \
                + (0.014506 \
                   + 4612.156534*T \
                   + 1.3915817*T**2 \
                   -0.00000044*T**3 \
                   -0.000029956*T**4 \
                   -0.0000000368*T**5)/15

        eps0 = 84381.406
        eps = eps0 \
              -46.836769*T \
              -0.0001831*T**2 \
              + 0.00200340*T**3 \
              -0.000000576*T**4 \
              -0.0000000434*T**5
        (F,D,Psi,del_psi,del_eps) = calc_nutation(T,nutation_series)

        Eps_ups = -del_psi*cos(eps) \
                  + 0.00264096*sin(Psi) \
                  + 0.00006352*sin(2*Psi) \
                  + 0.00001175*sin(2*F-2*D+3*Psi) \
                  + 0.00001121*sin(2*F-2*D+Psi) \
                  - 0.00000455*sin(2*F-2*D+2*Psi) \
                  + 0.00000202*sin(2*F+3*Psi) \
                  + 0.00000198*sin(2*F+Psi) \
                  - 0.00000172*sin(3*Psi) \
                  - 0.00000087*T*sin(Psi)

        GAST = GMST + Eps_ups/15

        psiA =  5038.481507*T \
                - 1.0790069*T**2 \
                - 0.00114045*T**3 \
                + 0.000132851*T**4 \
                - 0.0000000951*T**5
        omegaA = eps0 \
                 - 0.025754*T \
                 + 0.0512623*T**2 \
                 - 0.00772503*T**3 \
                 - 0.000000467*T**4 \
                 + 0.0000003337*T**5
        ChiA = 10.556403*T \
               - 2.3814292*T**2 \
               - 0.00121197*T**3 \
               + 0.000170663*T**4 \
               - 0.0000000560*T**5

        P = R3(-ChiA).dot(R1(-omegaA)).dot(R3(-psiA)).dot(R1(eps0))

        W = np.array([ [1, 0, -xp], [0, 1, yp], [xp, -yp, 1] ])

        S1 = sin(eps)
        S2 = sin(-del_psi)
        S3 = sin(-eps * -del_eps)
        C1 = cos(eps)
        C2 = cos(-del_eps)
        C3 = cos(-eps * -del_eps)
        N = np.array([ [C2, S2*C1, S2*S1],
                       [-S2*C3, C3*C2*C1-S1*S3, C3*C2*S1+C1*S3],
                       [S2*S3, -S3*C2*C1-S1*C3, -S3*C2*S1+C3*C1]])

        mas2rad = 1/206264806.247
        da0 = -14.6*mas2rad
        Xi0 = -16.6170*mas2rad
        eta0 = -6.8192*mas2rad
        B = np.array([ [1, da0, -Xi0], [-da0, 1, -eta0], [Xi0, eta0, 1]])

        R_ecef2eci = B.transpose().dot(P.transpose()).dot(N.transpose()).dot(R3(-GAST)).dot(W)

        return(R_ecef2eci.transpose())

def calc_nutation(T, nutation_series):
        # from http://aa.usno.navy.mil/publications/docs/Circular_179.pdf
        M = nutation_series[:,1:16]
        S =  nutation_series[:,15]
        S_dot = nutation_series[:,16]
        C_prime = nutation_series[:,17]
        C = nutation_series[:,18]
        C_dot = nutation_series[:,19]
        S_prime = nutation_series[:,20]

        rows = nutation_series.shape[0]
        cols = 14
        phi = np.zeros(14)
        phi[0] = 908103.259872 + 538101628.688982*T
        phi[1] = 655127.283060 + 210664136.433548*T
        phi[2] = 361679.244588 + 129597742.283429*T
        phi[3] = 1279558.798488 + 68905077.493988*T
        phi[4] = 123665.467464 + 10925660.377991*T
        phi[5] =  180278.799480 + 4399609.855732*T
        phi[6] =  1130598.018396 + 1542481.193933*T
        phi[7] = 1095655.195728 + 786550.320744*T
        phi[8] = 5028.8200*T + 1.112022*T**2
        phi[9] = 485868.249036 + 1717915923.2178*T + 31.8792*T**2 + 0.051635*T**3 - 0.00024470*T**4
        phi[10] = 1287104.79305 + 129596581.0481*T - 0.5532*T**2 + 0.000136*T**3 - 0.00001149*T**4
        phi[11] = 335779.526232 + 1739527262.8478*T - 12.7512*T**2 - 0.001037*T**3 + 0.00000417*T**4
        phi[12] = 1072260.70369 + 1602961601.2090*T - 6.3706*T**2 + 0.006593*T**3 - 0.00003169*T**4
        phi[13] = 450160.398036 - 6962890.5431*T + 7.4722*T**2 + 0.007702*T**3 - 0.00005939*T**4
        F = phi[11];
        D = phi[12];
        Psi = phi[13]
        del_psi = 0
        del_eps = 0
        for i in range(rows):
                Phi = 0
                for j in range(cols):
                        Phi = Phi + M[i,j]*phi[j]*phi[j]

                del_psi = del_psi + (S[i]+S_dot[i]*T)*sin(Phi) + C_prime[i]*cos(Phi)
                del_eps = del_eps + (C[i]+C_dot[i]*T)*cos(Phi) + S_prime[i]*sin(Phi)

        return(F,D,Psi,del_psi,del_eps)


def load_nutation_series(filename):
        nutation_series = np.genfromtxt(filename)
        return(nutation_series)

def R1(x):
        y = np.array([ [1, 0, 0], \
                       [0, math.cos(x), math.sin(x)], \
                       [0, -math.sin(x), math.cos(x)] ])
        return(y)

def R2(x):
        y = np.array([ [math.cos(x), 0, -math.sin(x)], \
                       [0, 1, 0], \
                       [math.sin(x), 0, math.cos(x)] ])
        return(y)

def R3(x):
        y = np.array([ [math.cos(x), math.sin(x), 0], \
                       [-math.sin(x), math.cos(x), 0], \
                       [0, 0, 1] ])
        return(y)

def unixtime2JD(UT1):
        return ( UT1 / 86400.0 ) + 2440587.5

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

	rPQW = np.array([[p*math.cos(nu)/(1 +e*math.cos(nu)), \
                          p*math.sin(nu)/(1+e*math.cos(nu)), \
                          0]])
        vPQW = np.array([[-math.sqrt(mu/p)*math.sin(nu), \
                          math.sqrt(mu/p)*(e+math.cos(nu)), \
                          0]])

	PQW2IJK = np.zeros((3,3));
        cO = math.cos(O)
        sO = math.sin(O)
        co = math.cos(o)
        so = math.sin(o)
        ci = math.cos(i)
        si = math.sin(i)
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
