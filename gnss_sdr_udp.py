import socket
import select
import gps_ephemeris_pb2  # protoc compiler
import monitor_pvt_pb2
import gnss_synchro_pb2
import datetime
import json
import math
from sortedcontainers import SortedDict
from itertools import islice
from gps_ephemeris_calc import posVelDtr
from mpmath import mpf, mp, sqrt, acos
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

EARTH_RADIUS = 6378  # in kilometers
IONOSPHERE_HEIGHT = 506.7  # in kilometers
ALPHA = 0.9782

# Mapping function for calculating VTEC from STEC
# https://link.springer.com/article/10.1007/s00190-023-01819-w
def mf(e):
    return math.sqrt(1 - (EARTH_RADIUS / (EARTH_RADIUS + IONOSPHERE_HEIGHT) * math.sin(ALPHA * e))**2)

mp.dps = 60
mp.pretty = False

SPEED_OF_LIGHT_M_S = mpf(299792458.0)

def closest(sorted_dict, key):
    "Return closest key in `sorted_dict` to given `key`."
    assert len(sorted_dict) > 0
    keys = list(islice(sorted_dict.irange(minimum=key), 1))
    keys.extend(islice(sorted_dict.irange(maximum=key, reverse=True), 1))
    return min(keys, key=lambda k: abs(key - k))


# Set up the UDP server
UDP_IP = "127.0.0.1"
EPH_PORT = 1237
PVT_PORT = 1238
SYN_PORT = 1239
LOG = True
SHOW_CLOCK = False

# Create a UDP socket
sock_eph = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_eph.bind((UDP_IP, EPH_PORT))
sock_pvt = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_pvt.bind((UDP_IP, PVT_PORT))
sock_syn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_syn.bind((UDP_IP, SYN_PORT))

ephemeri = SortedDict()
latest_ephemeri = {}
pvts = SortedDict()
obss = SortedDict()
prns = []

while True:
    ephemeri = SortedDict(latest_ephemeri)
    pvts = SortedDict()
    obss = SortedDict()
    prns = []

    for i in range(1000):
        ready, _, _ = select.select([sock_eph, sock_pvt, sock_syn], [], [])

        for sock in ready:
            if sock is sock_eph:
                eph_data, _ = sock.recvfrom(2048)
                #print(f"Received message from : {eph_data}")

                try:
                    ephemeris = gps_ephemeris_pb2.GpsEphemeris()
                    ephemeris.ParseFromString(eph_data[1:])  # G prefix

                    if ephemeris.PRN in ephemeri:
                        ephemeri[ephemeris.PRN][ephemeris.tow] = ephemeris
                    else:
                        ephemeri[ephemeris.PRN] = SortedDict({ephemeris.tow: ephemeris})
                    latest_ephemeri[ephemeris.PRN] = SortedDict({ephemeris.tow: ephemeris})

                except Exception as e:
                    print(f"Failed to decode message: {e}")

            if sock is sock_pvt:
                pvt_data, _ = sock.recvfrom(2048)

                try:
                    pvt = monitor_pvt_pb2.MonitorPvt()
                    pvt.ParseFromString(pvt_data)

                    pvts[pvt.rx_time] = {"clk_offset": pvt.user_clk_offset, "pos": [pvt.pos_x, pvt.pos_y, pvt.pos_z]}

                except Exception as e:
                    print(f"Failed to decode message: {e}")

            if sock is sock_syn:
                syn_data, _ = sock.recvfrom(2048)

                try:
                    stock = gnss_synchro_pb2.Observables()
                    stock.ParseFromString(syn_data)

                    for syn in stock.observable:
                        if syn.pseudorange_m == 0:
                            continue

                        if syn.prn not in prns:
                            prns.append(syn.prn)

                        #print(i)
                        if syn.prn in obss:
                            obss[syn.prn].append(syn)
                        else:
                            obss[syn.prn] = [syn]

                except Exception as e:
                    print(f"Failed to decode message: {e}")

    if SHOW_CLOCK:
        rx, offs = zip(*sorted(clock_offsets.items()))
        doffs = np.array([offs[i+1] - offs[i] for i in range(len(offs) - 1)])
        doffs = np.convolve(doffs, np.ones(100)/100, mode='valid')
        plt.plot(rx[:len(doffs)], doffs)
        plt.title("Clock drift")
        plt.xlabel("Temps écoulé (s)")
        plt.ylabel("Clock drift ($10^{-8}$ s/s)")
        plt.show()
        plt.title("Clock bias")
        plt.xlabel("Temps écoulé (s)")
        plt.ylabel("Clock bias (s)")
        plt.plot(rx, offs)
        plt.show()

    if not LOG:
        continue

    ionos = {}
    angles = {}
    pvts_ = {}
    pseudoranges = {}
    phases = {}
    satpos = {}
    for prn in prns:
        ionos[prn] = []
        angles[prn] = []
        pvts_[prn] = []
        pseudoranges[prn] = []
        phases[prn] = []
        satpos[prn] = []


    try:
        for prn in prns:
            for obs in obss[prn]:
                tow = obs.interp_tow_ms / 1000
                t = closest(pvts, obs.rx_time)
                clock_offset = mpf(pvts[t]["clk_offset"])
                ephemeris = ephemeri[prn][closest(ephemeri[prn], tow)]
                satx, saty, satz, _, _, _, dts = posVelDtr(tow, ephemeris)

                satpos[prn].append([satx, saty, satz, dts])

                g = (mpf(4213435), mpf(162752), mpf(4769685))
                s = (satx - g[0], saty - g[1], satz - g[2])
                
                gnorm = sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2])
                truerange = sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2])

                angle = acos((g[0]*s[0] + g[1]*s[1] + g[2]*s[2]) / (truerange * gnorm))
                angles[prn].append(angle)

                pseudoranges[prn].append(obs.pseudorange_m)

                #iono = mpf(obs.pseudorange_m) - truerange - SPEED_OF_LIGHT_M_S * (0 - dts)
                iono = mpf(obs.pseudorange_m) - truerange - SPEED_OF_LIGHT_M_S * (clock_offset - dts)
                #iono = (obs.pseudorange_m - obs.carrier_phase_rads * 2 / math.pi * (SPEED_OF_LIGHT_M_S / 1575420000))/2
                ionos[prn].append(iono)

                pvts_[prn].append({"clk_offset": float(clock_offset), "pos": pvts[t]["pos"]})

                phases[prn].append(obs.carrier_phase_rads)

            print(prn, sum(ionos[prn]) / len(ionos[prn]), sum(angles[prn]) / len(angles[prn]))

        date = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S")
        f = open(f"ionodata_20250312", "a")
        ionos = {prn: [float(vv) for vv in v] for prn, v in ionos.items()}
        angles = {prn: [float(vv) for vv in v] for prn, v in angles.items()}
        pseudoranges = {prn: [float(vv) for vv in v] for prn, v in pseudoranges.items()}
        phases = {prn: [float(vv) for vv in v] for prn, v in phases.items()}
        satpos = {prn: [(float(x), float(y), float(z), float(dts)) for (x, y, z, dts) in v] for prn, v in satpos.items()}
        f.write(json.dumps({"date": date, "iono_delays": ionos, "angles": angles, "pvt": pvts_, "pseudoranges": pseudoranges, "phases": phases, "satpos": satpos}))
        f.close()

    except Exception as e:
        print(e)
