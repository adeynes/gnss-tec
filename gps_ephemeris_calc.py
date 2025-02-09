import math
from mpmath import mpf, mp, sqrt, sin, cos, fmod, fabs, atan2

mp.dps = 60

GNSS_OMEGA_EARTH_DOT = mpf(7.292115e-5)#1467e-5)
GPS_GM = mpf(3.986005e14)
GPS_F = mpf(-4.442807633e-10)
GNSS_PI = mpf(3.1415926535898)
SPEED_OF_LIGHT_M_S = mpf(299792458.0)


def check_t(time):
    HALF_WEEK = mpf(302400.0)
    corrTime = time

    if time > HALF_WEEK:
        corrTime = time - mpf(2.0) * HALF_WEEK
    elif time < -HALF_WEEK:
        corrTime = time + mpf(2.0) * HALF_WEEK

    return corrTime


def posVelDtr(tt, eph):
    tt = mpf(tt)
    a = mpf(eph.sqrtA) * mpf(eph.sqrtA)
    n0 = sqrt(GPS_GM / (a*a*a))
    tk = check_t(tt - mpf(eph.toe))  # time from ephemeris reference epoch

    n = n0 + mpf(eph.delta_n)
    M = mpf(eph.M_0) + n*tk

    # Guess of eccentric anomaly
    E = M

    for _ in range(1000):
        E_old = E
        E = M + mpf(eph.ecc) * sin(E) # / (1.0 - mpf(eph.ecc) * cos(E))
        dE = fmod(E - E_old, 2.0 * GNSS_PI);

        if fabs(dE) < 1e-12:
            break

    sek = sin(E)
    cek = cos(E)
    OneMinusecosE = 1.0 - mpf(eph.ecc) * cek
    sq1e2 = sqrt(1.0 - mpf(eph.ecc) * mpf(eph.ecc))
    ekdot = n / OneMinusecosE

    tmp_Y = sq1e2 * sek
    tmp_X = cek - mpf(eph.ecc)
    nu = atan2(tmp_Y, tmp_X)

    phi = nu + mpf(eph.omega)

    s2pk = sin(2.0 * phi)
    c2pk = cos(2.0 * phi)
    pkdot = sq1e2 * ekdot / OneMinusecosE

    u = phi + mpf(eph.Cuc) * c2pk + mpf(eph.Cus) * s2pk
    suk = sin(u)
    cuk = cos(u)
    ukdot = pkdot * (1.0 + 2.0 * (mpf(eph.Cus) * c2pk - mpf(eph.Cuc) * s2pk))

    r = a * OneMinusecosE + mpf(eph.Crc) * c2pk + mpf(eph.Crs) * s2pk
    rkdot = a * mpf(eph.ecc) * sek * ekdot + 2.0 * pkdot * (mpf(eph.Crs) * c2pk - mpf(eph.Crc) * s2pk)

    i = mpf(eph.i_0) + mpf(eph.idot) * tk + mpf(eph.Cic) * c2pk + mpf(eph.Cis) * s2pk
    sik = sin(i)
    cik = cos(i)
    ikdot = mpf(eph.idot) + 2.0 * pkdot * (mpf(eph.Cis) * c2pk - mpf(eph.Cic) * s2pk)

    Omega_dot = mpf(eph.OMEGAdot) - GNSS_OMEGA_EARTH_DOT
    Omega = mpf(eph.OMEGA_0) + Omega_dot * tk - GNSS_OMEGA_EARTH_DOT * mpf(eph.toe)
    sok = sin(Omega)
    cok = cos(Omega)

    xprime = r * cuk
    yprime = r * suk

    pos0 = xprime * cok - yprime * cik * sok
    pos1 = xprime * sok + yprime * cik * cok
    pos2 = yprime * sik

    xpkdot = rkdot * cuk - yprime * ukdot
    ypkdot = rkdot * suk + xprime * ukdot
    tmp = ypkdot * cik - pos2 * ikdot

    vel0 = -Omega_dot * pos1 + xpkdot * cok - tmp * sok
    vel1 = Omega_dot * pos0 + xpkdot * sok + tmp * cok
    vel2 = yprime * cik * ikdot + ypkdot * sik

    tk = check_t(tt - mpf(eph.toc))  # time from ephemeris reference clock

    dtr = mpf(eph.af0) + mpf(eph.af1) * tk + mpf(eph.af2) * tk * tk
    dtr -= 2.0 * sqrt(GPS_GM * a) * mpf(eph.ecc) * sek / (SPEED_OF_LIGHT_M_S * SPEED_OF_LIGHT_M_S)

    return pos0, pos1, pos2, vel0, vel1, vel2, dtr


# eph2pos, TODO HAS correction
def posDtr2(tt, eph):
    a = eph.sqrtA * eph.sqrtA
    n0 = math.sqrt(GPS_GM / (a*a*a))
    tk = check_t(tt - eph.toe)  # time from ephemeris reference epoch

    n = n0 + eph.delta_n
    M = eph.M_0 + n*tk

    # Guess of eccentric anomaly
    E = M

    for _ in range(20):
        Ek = E
        E -= (E - eph.ecc * math.sin(E) - M) / (1.0 - eph.ecc * math.cos(E))
        dE = math.fmod(E - E_old, 2.0 * GNSS_PI);

        if abs(dE) < 1e-12:
            break;

    sinE = math.sin(E)
    cosE = math.cos(E)

    u = math.atan2(math.sqrt(1.0 - eph.ecc * eph.ecc) * sinE, cosE - eph.ecc) + eph.omega
    r = a * (1.0 - eph.ecc * cosE)
    i = eph.i_0 + eph.idot * tk
    sin2u = math.sin(2.0 * u)
    cos2u = math.cos(2.0 * u)
    u += eph.Cus * sin2u + eph.Cuc * cos2u
    r += eph.Crs * sin2u + eph.Crc * cos2u
    i += eph.Cis * sin2u + eph.Cic * cos2u
    x = r * math.cos(u)
    y = r * math.sin(u)
    cosi = math.cos(i)

    O = eph.OMEGA_0 + (eph.OMEGAdot - GNSS_OMEGA_EARTH_DOT) * tk - GNSS_OMEGA_EARTH_DOT * eph.toe
    sinO = math.sin(O)
    cosO = math.cos(O)
    rs = (x * cosO - y * cosi * sinO, x * sinO + y * cosi * cosO, y * math.sin(i))

    tk = check_t(tt - eph.toc)
    dtr = eph.af0 + eph.af1 * tk + eph.af2 * tk * tk
    dtr -= 2.0 * math.sqrt(GPS_GM * a) * eph.ecc * sinE / (SPEED_OF_LIGHT_M_S * SPEED_OF_LIGHT_M_S)

    return rs, dtr
