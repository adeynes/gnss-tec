import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import math
from math import sqrt
import json
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.optimize import minimize  # For least squares optimization
from scipy.stats import linregress
import pandas as pd
from io import StringIO

# Constants
EARTH_RADIUS = 6378  # in kilometers
IONOSPHERE_HEIGHT = 506.7  # in kilometers
ALPHA = 0.9782
SPEED_OF_LIGHT_M_S = 299792458.0
ALIGN_ON = "18"  # The PRN we align to
SMOOTHING = 20

MULT_FACTOR = math.pi

TEC_SHIFT = 300
Y_CORR = 162777

# Mapping function for calculating VTEC from STEC
# https://link.springer.com/article/10.1007/s00190-023-01819-w
def mf_2(a):
    return math.sqrt(1 - (EARTH_RADIUS / (EARTH_RADIUS + IONOSPHERE_HEIGHT) * math.sin(ALPHA * a))**2)

def mf(a):
    return mf_2(a)

def mf_trop(a):
    return 1.001/math.sqrt(0.002001 + math.sin(math.pi/2 - a)**2)

P0 = 1015.75 + 1/5 * (1011.75 - 1015.75)
dP = -2.25 + 1/5 * (1.75 - 2.25)
T0 = 283.15 + 1/5 * (272.15 - 283.15)
dT = 11 + 1/5 * (15 - 11)
e0 = 11.66 + 1/5 * (6.78 - 11.66)
de = 7.24 + 1/5 * (5.36 - 7.24)
b0 = 0.00558 + 1/5 * (0.00539 - 0.00558)
db = 0.00032 + 1/5 * (0.00081 - 0.00032)
l0 = 2.57 + 1/5 * (1.81 - 2.57)
dl = 0.46 + 1/5 * (0.74 - 0.46)

q = math.cos(2*math.pi * (21 - 28) / 365.25)

P = P0 - q*dP
T = T0 - q*dT
e = e0 - q*de
b = b0 - q*db
l = l0 - q*dl

k1 = 77.604
k2 = 382000
Rd = 287.054
gm = 9.784
g = 9.80665
H = 158

Tdry = (1 - b*H/T) ** (g/(Rd*b)) * 1e-6 * k1 * Rd * P / gm
Twet = (1 - b*H/T) ** ((l+1)*g/(Rd*b) - 1) * 1e-6 * k2 * Rd / ((l+1)*gm - b*Rd) * e / T
print(Twet+Tdry)

# Load and parse the concatenated JSON blocks from the file
file_path = 'ionodata_20250120'

def load_data(file_path):
    with open(file_path, 'r') as file:
        raw_data = file.read()
    # Split the concatenated JSON blocks
    json_blocks = raw_data.replace('}{', '}\n{').split('\n')
    return [json.loads(block) for block in json_blocks]

data_blocks = load_data(file_path)

# Initialize storage for plotting
prn_data = {}

# Parse the data
for block in data_blocks:
    timestamp = datetime.strptime(block["date"], "%Y%m%dT%H%M%S")
 
    for prn, stecs in block["stecs"].items():
        iono_delays = block["iono_delays"].get(prn, [])
        angles = block["angles"].get(prn, [])
        satpos = block["satpos"].get(prn, [])
        pseudoranges = block["pseudoranges"].get(prn, [])
        clk_offset = block["clk_offset"].get(prn, [])

        if prn not in prn_data:
            prn_data[prn] = {"timestamps": [], "tecs": [], "angles": []}

        # Compute vertical TEC (VTEC) from STEC and angle using the mapping function
        vtecs = []
        for iono, sat, c, p, a in zip(iono_delays, satpos, clk_offset, pseudoranges, angles):
            g = (4213435, 162752, 4769685)
            [satx, saty, satz] = sat
            s = (satx - g[0], saty - g[1], satz - g[2])
                
            truerange = sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2])

            dts = 1/SPEED_OF_LIGHT_M_S * (iono - p + truerange) + c

            newg = (4213435, Y_CORR, 4769685)
            news = (satx - newg[0], saty - newg[1], satz - newg[2])
            newrange = sqrt(news[0]*news[0] + news[1]*news[1] + news[2]*news[2])
            dtrplusiono = 1575420000**2 / (40.3e16) / MULT_FACTOR * ((p - newrange) + SPEED_OF_LIGHT_M_S*dts - (Twet + Tdry)/mf_trop(a))
            tec = dtrplusiono

            #tec = 1575420000**2 / (40.3e16) * (iono + SPEED_OF_LIGHT_M_S * c)
            if abs(tec - 7e6/MULT_FACTOR) > 1e6/MULT_FACTOR:
                continue
            vtecs.append(tec)

        if vtecs:
            prn_data[prn]["timestamps"].append(timestamp)
            prn_data[prn]["tecs"].append(sum(vtecs)/len(vtecs))
            prn_data[prn]["angles"].append(sum(angles)/len(angles))

# Function for smoothing using a simple moving average
def smooth_data(values, window_size=SMOOTHING):
    if len(values) < window_size:
        window_size = len(values)
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid')

# Smooth the data for all PRNs
smoothed_prn_data = {}
for prn, data in prn_data.items():
    smoothed_values = smooth_data(data["tecs"], SMOOTHING)
    smoothed_angles = smooth_data(data["angles"], SMOOTHING)
    smoothed_timestamps = data["timestamps"][:len(smoothed_values)]
    smoothed_prn_data[prn] = {"timestamps": smoothed_timestamps, "tecs": smoothed_values, "angles": smoothed_angles}


# Align the data
raw_data, smoothed_data = prn_data, smoothed_prn_data


def remove_large_slopes_datetime(x, y, z, max_slope):
    """
    Removes points from x and y where the slope exceeds max_slope.
    
    Parameters:
        x (array-like): Independent variable points (datetime).
        y (array-like): Dependent variable points (numeric values).
        max_slope (float): Maximum allowed slope. Points with a larger slope are removed.
    
    Returns:
        tuple: Filtered x and y arrays.
    """
    # Convert datetime to seconds (relative to the first point)
    x_seconds = np.array([(t - x[0]).total_seconds() for t in x])
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)
    
    # Calculate slopes
    slopes = np.abs(np.diff(y) / np.diff(x_seconds))
    
    # Create a mask for points where the slope is within the allowed range
    mask = slopes <= max_slope
    
    # Include the last valid point by appending True to the mask
    valid_indices = np.hstack(([True], mask)) & np.hstack((mask, [True]))
    
    # Filter x and y using the mask
    filtered_x = [x[i] for i in range(len(x)) if valid_indices[i]]
    filtered_y = [y[i] for i in range(len(y)) if valid_indices[i]]
    filtered_z = [z[i] for i in range(len(z)) if valid_indices[i]]
    
    return filtered_x, filtered_y, filtered_z

def add_gaps(x, y, z, max_gap=timedelta(minutes=2)):
    """
    Inserts NaT values into datetime x and NaN into y where gaps in x exceed max_gap.
    """
    x = np.array(x, dtype='datetime64[us]')  # Ensure x is numpy datetime64
    y = np.array(y, dtype=float)  # Ensure y is a float array
    z = np.array(z, dtype=float)
    
    # Find gaps in x
    gaps = np.diff(x) > max_gap
    if not np.any(gaps):
        return x, y, z  # No gaps to handle
    
    # Split and insert NaT where gaps occur
    new_x = []
    new_y = []
    new_z = []
    for i in range(len(x) - 1):
        new_x.append(x[i])
        new_y.append(y[i])
        new_z.append(z[i])
        if gaps[i]:  # Add NaT and NaN if there's a gap
            new_x.append(np.datetime64('NaT'))
            new_y.append(np.nan)
            new_z.append(np.nan)
    # Append the last point
    new_x.append(x[-1])
    new_y.append(y[-1])
    new_z.append(z[-1])
    
    return np.array(new_x), np.array(new_y), np.array(new_z)


def linear_regression_all(data):
    """
    Perform a single linear regression on data from all satellites, combining all points.
    
    Parameters:
        data (dict): Dictionary with satellite IDs as keys and a dictionary with:
                     - "timestamps": list of datetime objects
                     - "tecs": list of numeric values
    
    Returns:
        dict: Regression results including slope, intercept, and statistical metrics.
    """
    all_x = []
    all_y = []

    # Collect all points, ignoring gaps (None, NaT, or NaN values)
    for prn, satellite_data in data.items():
        x = satellite_data["timestamps"]
        y = satellite_data["tecs"]
        for xi, yi in zip(x, y):
            if xi is not None and not pd.isna(xi) and not pd.isna(yi):  # Check for None, NaT, and NaN values
                all_x.append(xi)
                all_y.append(yi)

    # Sort by time to ensure proper time alignment
    sorted_indices = np.argsort(all_x)
    all_x = np.array(all_x)[sorted_indices]
    all_y = np.array(all_y)[sorted_indices]

    # Convert datetime to seconds relative to the first timestamp
    # Only include timestamps that are not None or NaT
    all_x_seconds = np.array([(t - all_x[0]).item().total_seconds() for t in all_x if t is not None and not pd.isna(t)])

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(all_x_seconds, all_y)
    slope -= 0.00005

    # Perform subtraction of regression from the data
    residuals = {}
    for prn, satellite_data in data.items():
        x = satellite_data["timestamps"]
        y = satellite_data["tecs"]
        
        # Filter out None and NaT timestamps before calculating seconds
        #valid_x = [t for t in x if t is not None and not pd.isna(t)]
        #valid_y = [y_i for t, y_i in zip(x, y) if t is not None and not pd.isna(t)]
        
        # Convert valid x to seconds
        x_seconds = np.array([(t - all_x[0]).item().total_seconds() for t in x])

        # Calculate predicted y values based on the regression
        y_pred = slope * x_seconds + intercept

        # Ensure y and y_pred have the same length before subtraction
        if len(y) == len(y_pred):
            residuals[prn] = {"timestamps": x, "stecs": np.array(y) - y_pred, "angles": satellite_data["angles"]}
        else:
            print(f"Length mismatch for PRN {prn}: valid_y length {len(valid_y)}, y_pred length {len(y_pred)}")


    # Create x_fit values for the regression line, in seconds
    x_fit = np.linspace(0, all_x_seconds[-1], 100)
    y_fit = slope * x_fit + intercept

    # Convert x_fit from seconds to datetime by adding timedelta to the first timestamp
    all_x_datetime = all_x[0].tolist()  # This converts it to a Python `datetime`
    x_fit_datetimes = [all_x_datetime + timedelta(seconds=xf) for xf in x_fit]

    # Plot the regression line
    plt.plot(x_fit_datetimes, y_fit, label='Regression Line', color='red')


    # Return regression results and residuals
    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'residuals': residuals
    }


def shifted_tecs_by_time(shifts, res, prn_i):
    print(shifts)
    tecs_by_time = {}

    for prn, data in res.items():
        shifted_stecs = data["stecs"] + shifts[prn_i[prn]]
        tecs = np.array(shifted_stecs) * np.array([mf(a) for a in data["angles"]])
        for t, tec in zip(data["timestamps"], tecs):
            if t not in tecs_by_time:
                tecs_by_time[t] = [tec]
            else:
                tecs_by_time[t].append(tec)

    return tecs_by_time

def shifted_tecs(shifts, res, prn_i):
    tecss = {}

    for prn, data in res.items():
        shifted_stecs = data["stecs"] + shifts[prn_i[prn]]
        tecs = np.array(shifted_stecs) * np.array([mf(a) for a in data["angles"]])
        tecss[prn] = {"timestamps": data["timestamps"], "tecs": tecs}

    return tecss


def average_stdev(shifts, res, prn_i):
    return np.mean([np.std(tecs) for _, tecs in shifted_tecs_by_time(shifts, res, prn_i).items()])


# Plotting setup
bright_colors = [
    "#FF0000",  # Red
    "#0000D0",  # Blue
    "#A000A0",  # Purple
    "#00D000",  # Bright Green
    "#FF00FF",  # Magenta
    "#00D0FF",  # Cyan
    "#A52A2A",  # Brown
    "#FFAA00",  # Orange
    "#008080",  # Teal
    "#009400"   # Dark Green
]

# Create a colormap
colormap = ListedColormap(bright_colors, name="bright_cmap")
num_colors = 10
num_prns = len(prn_data)
colors = [colormap((i % num_colors) / num_colors) for i in range(num_prns)]  # Generate unique colors for each PRN
line_styles = ['solid', 'dashed', 'dashdot', 'dotted']

# Plot the data
plt.figure(figsize=(10, 6))

regdata = {}
prn_i = {}

i = 0

# Handle large slopes and add gaps
for prn, _ in raw_data.items():
    if len(smoothed_data[prn]["tecs"]) == 1:
        continue

    prn_i[prn] = i
    i += 1
    
    # Remove large slopes and add gaps
    times, tecs, angles = remove_large_slopes_datetime(smoothed_data[prn]["timestamps"], smoothed_data[prn]["tecs"], smoothed_data[prn]["angles"], 1/6/MULT_FACTOR)
    times, tecs, angles = add_gaps(times, tecs, angles)
    
    # Remove None or NaT timestamps from the times list
    valid_times = [t for t in times if not pd.isna(t)]
    valid_tecs = [tecs[i] for i, t in enumerate(times) if not pd.isna(times[i])]
    valid_angles = [angles[i] for i, t in enumerate(times) if not pd.isna(times[i])]

    plt.plot(times, tecs)
    
    regdata[prn] = {"timestamps": valid_times, "tecs": valid_tecs, "angles": valid_angles}

# Now perform linear regression on the adjusted data
reg = linear_regression_all(regdata)
plt.show()

initial_shifts = np.array([309.12312175, 307.17449878, 288.70317809, 310.84991494, 318.69896499,
 326.15650953, 313.97817739, 321.01894933, 321.83201357, 302.61413321,
 309.99201226, 316.18385416, 321.68845462, 295.36260896, 301.84653261,
 318.7469365,  296.66161373, 334.40472387, 354.21426591, 338.3159859,
 297.46213385, 289.08206429, 312.97422112, 301.93458858, 313.7177469,
 308.05101559, 306.19244922, 317.81444355, 317.40306583, 291.32456312,
 314.88610115, 283.40447019]) / np.array([MULT_FACTOR]*len(prn_i))

#initial_shifts = [TEC_SHIFT]*len(prn_i)

result = minimize(fun=average_stdev, x0=initial_shifts, args=(reg["residuals"], prn_i), tol=0.01)

# Print the optimized shifts
shifts = result.x
#shifts = initial_shifts
print(shifts)
print(result.fun)

j = 0
print(prn_i)

avg_tecs = {}

# Second loop: subtract regression and adjust by mf(angle)
#for prn, data in reg["residuals"].items():
for prn, data in shifted_tecs(shifts, reg["residuals"], prn_i).items():
    print(j)
    times = data["timestamps"]
    tecs = [t - 60*math.pi/MULT_FACTOR for t in data["tecs"]]
    angles = reg["residuals"][prn]["angles"]

    ptimes, ptecs, pangles = add_gaps(times, tecs, angles)

    for i in range(len(ptimes)):
        if ptimes[i] not in avg_tecs:
            avg_tecs[ptimes[i]] = [ptecs[i]]
        else:
            avg_tecs[ptimes[i]].append(ptecs[i])

    #ptecs = [(ptec + TEC_SHIFT) * mf(a) for ptec, a in zip(ptecs, pangles)]

    # Plot the adjusted data
    plt.plot(ptimes, ptecs, label=f'PRN {prn}', color=colors[j], linestyle=line_styles[j // num_colors], linewidth=1, alpha=0.5)
    #plt.plot(ptimes, [180*a for a in pangles], color=colors[j], linestyle=line_styles[j // num_colors])

    j += 1

for i in avg_tecs.keys():
    avg_tecs[i] = sum(avg_tecs[i])/len(avg_tecs[i])

plt.plot(avg_tecs.keys(), avg_tecs.values())


dourbes = """
Time                     CS   TEC QD
2025-01-20T12:00:01.000Z  85  14.6 //
2025-01-20T12:05:02.000Z  75  37.8 //
2025-01-20T12:10:00.000Z  80  27.0 //
2025-01-20T12:15:01.000Z  95  24.9 //
2025-01-20T12:20:02.000Z  90  19.4 //
2025-01-20T12:25:00.000Z  80  21.7 //
2025-01-20T12:30:01.000Z 100  23.7 //
2025-01-20T12:35:02.000Z  95  21.8 //
2025-01-20T12:40:00.000Z  95  26.7 //
2025-01-20T12:45:01.000Z  80  30.1 //
2025-01-20T12:50:02.000Z  80  30.5 //
2025-01-20T12:55:00.000Z  70  13.9 //
2025-01-20T13:00:01.000Z  90  19.0 //
2025-01-20T13:05:02.000Z  65  25.5 //
2025-01-20T13:15:01.000Z 100  27.2 //
2025-01-20T13:20:02.000Z 100  24.7 //
2025-01-20T13:25:00.000Z  80  24.2 //
2025-01-20T13:30:01.000Z  95  23.1 //
2025-01-20T13:35:02.000Z  95  20.8 //
2025-01-20T14:55:00.000Z  45   4.6 //
2025-01-20T15:00:01.000Z  40   2.9 //
2025-01-20T15:10:00.000Z  70   2.7 //
2025-01-20T15:15:01.000Z 100  27.6 //
2025-01-20T15:20:02.000Z  95  24.1 //
2025-01-20T15:25:00.000Z  95  19.6 //
2025-01-20T15:30:01.000Z  95  21.6 //
2025-01-20T15:35:02.000Z  95  26.2 //
2025-01-20T15:40:00.000Z  95  24.3 //
2025-01-20T15:45:01.000Z  95  21.8 //
2025-01-20T15:50:02.000Z  95  17.6 //
2025-01-20T15:55:00.000Z  95  21.6 //
2025-01-20T16:00:01.000Z  95  23.0 //
2025-01-20T16:05:02.000Z  85  17.8 //
2025-01-20T16:10:00.000Z  95  15.3 //
2025-01-20T16:15:01.000Z  95  15.4 //
2025-01-20T16:20:02.000Z  95  13.9 //
2025-01-20T16:25:00.000Z  95  15.1 //
2025-01-20T16:30:01.000Z  95  13.2 //
2025-01-20T16:35:02.000Z  95  11.3 //
2025-01-20T16:40:00.000Z  85  12.4 //
2025-01-20T16:45:01.000Z  85  13.9 //
2025-01-20T16:50:02.000Z  80  13.6 //
2025-01-20T16:55:00.000Z  95  16.1 //
2025-01-20T17:00:01.000Z  90   8.3 //
2025-01-20T17:05:02.000Z  65   5.9 //
2025-01-20T17:10:00.000Z  80   8.1 //
2025-01-20T17:15:01.000Z  80   5.6 //
2025-01-20T17:20:02.000Z  80   4.6 //
2025-01-20T17:25:00.000Z  80   3.0 //
2025-01-20T17:30:01.000Z  85   5.2 //
2025-01-20T17:35:02.000Z  65   2.6 //
2025-01-20T17:40:00.000Z  45   2.8 //
2025-01-20T17:45:01.000Z  45   4.9 //
2025-01-20T17:55:00.000Z  40   3.7 //
2025-01-20T18:05:02.000Z  25   3.5 //
2025-01-20T18:10:00.000Z  45   0.4 //
2025-01-20T18:15:01.000Z  70   5.8 //
2025-01-20T18:30:01.000Z  90   5.2 //
2025-01-20T18:35:02.000Z  90   6.4 //
2025-01-20T18:40:00.000Z  90   5.3 //
2025-01-20T18:45:01.000Z  90   5.5 //
2025-01-20T18:50:02.000Z  90   4.0 //
2025-01-20T18:55:00.000Z  90   5.4 //
2025-01-20T19:00:01.000Z  75   4.9 //
2025-01-20T19:05:02.000Z  95   3.9 //
2025-01-20T19:10:00.000Z  90   3.3 //
2025-01-20T19:15:01.000Z  95   3.1 //
2025-01-20T19:20:02.000Z  90   4.1 //
2025-01-20T19:25:00.000Z  95   3.1 //
2025-01-20T19:35:02.000Z 100   3.9 //
2025-01-20T19:45:01.000Z 100   2.9 //
2025-01-20T19:50:02.000Z 100   2.9 //
2025-01-20T19:55:00.000Z  95   3.7 //
2025-01-20T20:00:01.000Z  95   1.9 //
2025-01-20T20:05:02.000Z  95   3.6 //
2025-01-20T20:15:01.000Z  90   4.3 //
2025-01-20T20:20:02.000Z 100   1.3 //
2025-01-20T20:25:00.000Z 100   2.7 //
2025-01-20T20:30:01.000Z  70   3.1 //
2025-01-20T20:35:02.000Z  90   3.7 //
2025-01-20T20:40:00.000Z  85   1.7 //
2025-01-20T20:45:01.000Z  90   2.1 //
2025-01-20T20:50:02.000Z  95   3.3 //
2025-01-20T20:55:00.000Z 100   3.0 //
2025-01-20T21:00:01.000Z 100   3.2 //
2025-01-20T21:05:02.000Z 100   2.6 //
2025-01-20T21:10:02.000Z  95   2.7 //
2025-01-20T21:15:01.000Z  95   3.6 //
2025-01-20T21:20:02.000Z 100   1.8 //
2025-01-20T21:25:02.000Z  95   2.2 //
2025-01-20T21:30:01.000Z 100   3.2 //
2025-01-20T21:35:02.000Z  95   4.0 //
2025-01-20T21:40:02.000Z  95   2.4 //
2025-01-20T21:45:01.000Z 100   2.6 //
2025-01-20T21:50:02.000Z 100   1.9 //
2025-01-20T21:55:02.000Z  85   4.3 //
2025-01-20T22:00:01.000Z  95   4.0 //
2025-01-20T22:05:02.000Z 100   2.5 //
2025-01-20T22:10:02.000Z 100   3.3 //
2025-01-20T22:15:01.000Z 100   3.2 //
2025-01-20T22:20:02.000Z 100   3.0 //
2025-01-20T22:25:02.000Z  95   2.8 //
2025-01-20T22:30:01.000Z 100   3.3 //
2025-01-20T22:35:02.000Z 100   3.4 //
2025-01-20T22:40:02.000Z  95   3.4 //
2025-01-20T22:45:01.000Z  95   2.8 //
2025-01-20T22:50:02.000Z 100   2.3 //
2025-01-20T22:55:02.000Z 100   2.9 //
2025-01-20T23:00:01.000Z 100   2.4 //
2025-01-20T23:05:02.000Z 100   3.3 //
2025-01-20T23:10:02.000Z 100   3.2 //
2025-01-20T23:15:01.000Z  95   3.9 //
2025-01-20T23:20:02.000Z  95   3.0 //
2025-01-20T23:25:02.000Z  95   2.6 //
2025-01-20T23:30:01.000Z  95   2.7 //
2025-01-20T23:35:02.000Z 100   3.4 //
2025-01-20T23:40:02.000Z 100   2.7 //
2025-01-20T23:45:01.000Z 100   3.1 //
2025-01-20T23:50:02.000Z  95   4.5 //
2025-01-20T23:55:02.000Z  95   4.7 //
2025-01-21T00:00:01.000Z 100   3.5 //
2025-01-21T00:05:02.000Z  95   3.4 //
2025-01-21T00:10:02.000Z 100   3.9 //
2025-01-21T00:15:01.000Z  95   4.4 //
2025-01-21T00:20:02.000Z  95   3.4 //
2025-01-21T00:25:02.000Z  95   3.9 //
2025-01-21T00:30:01.000Z  70   3.1 //
2025-01-21T00:35:02.000Z  70   2.9 //
2025-01-21T00:40:02.000Z  70   3.1 //
2025-01-21T00:45:01.000Z  70   2.8 //
2025-01-21T00:50:02.000Z  75   3.8 //
2025-01-21T00:55:02.000Z  75   3.7 //
2025-01-21T01:00:01.000Z  70   2.9 //
2025-01-21T01:05:02.000Z  65   3.5 //
2025-01-21T01:10:02.000Z  70   2.7 //
2025-01-21T01:15:01.000Z  75   2.8 //
2025-01-21T01:20:02.000Z  75   3.0 //
2025-01-21T01:25:02.000Z  70   3.6 //
2025-01-21T01:30:01.000Z  65   3.8 //
2025-01-21T01:35:02.000Z  70   3.7 //
2025-01-21T01:40:02.000Z  70   3.4 //
2025-01-21T01:45:01.000Z  70   3.4 //
2025-01-21T01:50:02.000Z  65   3.6 //
2025-01-21T01:55:02.000Z  70   2.8 //
2025-01-21T02:00:01.000Z  70   3.4 //
2025-01-21T02:05:02.000Z  70   3.2 //
2025-01-21T02:10:02.000Z  65   2.1 //
2025-01-21T02:15:01.000Z  65   3.0 //
2025-01-21T02:20:02.000Z  65   3.2 //
2025-01-21T02:25:02.000Z  75   3.2 //
2025-01-21T02:30:01.000Z  65   3.4 //
2025-01-21T02:35:02.000Z  90   3.7 //
2025-01-21T02:40:02.000Z  90   3.3 //
2025-01-21T02:45:01.000Z  90   3.9 //
2025-01-21T02:50:02.000Z  90   2.8 //
2025-01-21T02:55:02.000Z  90   3.3 //
2025-01-21T03:00:01.000Z  90   3.3 //
2025-01-21T03:05:02.000Z  80   2.7 //
2025-01-21T03:10:02.000Z  60   2.7 //
2025-01-21T03:15:01.000Z  90   3.9 //
2025-01-21T03:20:02.000Z  60   2.4 //
2025-01-21T03:25:02.000Z  80   2.3 //
2025-01-21T03:30:01.000Z  60   0.9 //
2025-01-21T03:35:02.000Z  80   1.5 //
2025-01-21T03:40:02.000Z  80   2.6 //
2025-01-21T03:45:01.000Z  80   2.2 //
2025-01-21T03:50:02.000Z  60   2.0 //
2025-01-21T03:55:02.000Z  45   3.1 //
2025-01-21T04:00:01.000Z  90   2.3 //
2025-01-21T04:05:02.000Z  70   2.2 //
2025-01-21T04:10:02.000Z  50   1.4 //
2025-01-21T04:15:01.000Z  40   1.7 //
2025-01-21T04:20:02.000Z  70   2.6 //
2025-01-21T04:25:02.000Z  40   1.6 //
2025-01-21T04:30:01.000Z  80   2.5 //
2025-01-21T04:35:02.000Z  50   2.5 //
2025-01-21T04:40:02.000Z  70   1.0 //
2025-01-21T04:45:01.000Z  40   0.8 //
2025-01-21T04:50:02.000Z  20   0.8 //
2025-01-21T04:55:02.000Z  80   2.8 //
2025-01-21T05:00:01.000Z  80   2.3 //
2025-01-21T05:05:02.000Z  80   1.6 //
2025-01-21T05:10:02.000Z  80   2.2 //
2025-01-21T05:15:01.000Z  95   2.5 //
2025-01-21T05:20:02.000Z  80   1.9 //
2025-01-21T05:25:02.000Z  80   1.9 //
2025-01-21T05:30:01.000Z  90   1.4 //
2025-01-21T05:35:08.000Z  80   1.0 //
2025-01-21T05:40:00.000Z  80   2.4 //
2025-01-21T05:45:01.000Z  80   1.0 //
2025-01-21T05:50:08.000Z  60   0.7 //
2025-01-21T05:55:00.000Z  40   2.6 //
2025-01-21T06:00:01.000Z  90   3.5 //
2025-01-21T06:05:08.000Z  90   2.8 //
2025-01-21T06:10:00.000Z  90   2.9 //
2025-01-21T06:15:01.000Z  85   2.3 //
2025-01-21T06:20:08.000Z  80   2.5 //
2025-01-21T06:25:00.000Z  90   3.3 //
2025-01-21T06:30:01.000Z  95   2.5 //
2025-01-21T06:35:02.000Z  80   2.7 //
2025-01-21T06:40:00.000Z  90   1.3 //
2025-01-21T06:45:01.000Z  75   0.9 //
2025-01-21T06:50:02.000Z  95   3.2 //
2025-01-21T06:55:00.000Z  95   4.0 //
2025-01-21T07:00:01.000Z  80   3.6 //
2025-01-21T07:05:02.000Z  90   4.4 //
2025-01-21T07:10:00.000Z  95   5.0 //
2025-01-21T07:15:01.000Z  90   6.6 //
2025-01-21T07:20:02.000Z 100   6.8 //
2025-01-21T07:25:00.000Z  95   9.0 //
2025-01-21T07:30:01.000Z  90   9.6 //
2025-01-21T07:35:02.000Z  95   8.8 //
2025-01-21T07:40:00.000Z  80   7.6 //
2025-01-21T07:55:00.000Z  95   9.3 //
2025-01-21T08:00:01.000Z  95   9.3 //
2025-01-21T08:05:02.000Z 100  11.3 //
2025-01-21T08:10:00.000Z 100  15.4 //
2025-01-21T08:15:01.000Z  95  15.7 //
2025-01-21T08:20:02.000Z  90  15.4 //
2025-01-21T08:25:00.000Z 100  17.4 //
2025-01-21T08:30:01.000Z  90  17.5 //
2025-01-21T08:35:02.000Z  95  17.6 //
2025-01-21T08:40:00.000Z  80  15.2 //
2025-01-21T08:45:01.000Z  95  16.2 //
2025-01-21T08:50:02.000Z  95  21.5 //
2025-01-21T08:55:00.000Z 100  25.0 //
2025-01-21T09:00:01.000Z  70  21.5 //
2025-01-21T09:10:00.000Z  95  20.5 //
2025-01-21T09:15:01.000Z  80  20.4 //
2025-01-21T09:20:02.000Z  85  21.5 //
2025-01-21T09:25:00.000Z  80  22.6 //
2025-01-21T09:30:01.000Z  90  17.4 //
2025-01-21T09:35:02.000Z  95  17.5 //
2025-01-21T09:40:00.000Z  95  13.4 //
2025-01-21T09:45:01.000Z  80  11.2 //
2025-01-21T09:50:02.000Z  70   9.0 //
2025-01-21T09:55:00.000Z  50   9.1 //
2025-01-21T10:00:01.000Z  60  21.7 //
2025-01-21T10:05:02.000Z  90  18.2 //
2025-01-21T10:10:00.000Z  90  18.3 //
2025-01-21T10:15:01.000Z  95  19.4 //
2025-01-21T10:20:02.000Z  85  18.6 //
2025-01-21T10:25:00.000Z  80  26.2 //
2025-01-21T10:30:01.000Z  70  36.0 //
2025-01-21T10:40:00.000Z  95  24.8 //
2025-01-21T10:45:01.000Z 100  25.9 //
2025-01-21T10:50:02.000Z  95  30.3 //
2025-01-21T10:55:00.000Z  80  29.1 //
2025-01-21T11:00:01.000Z  85  27.2 //
2025-01-21T11:05:02.000Z  95  29.0 //
2025-01-21T11:10:00.000Z  90  23.8 //
2025-01-21T11:15:01.000Z  95  23.7 //
2025-01-21T11:20:02.000Z  95  22.2 //
2025-01-21T11:25:00.000Z  75  21.6 //
2025-01-21T11:30:01.000Z  95  23.0 //
2025-01-21T11:35:02.000Z  80  24.8 //
2025-01-21T11:40:00.000Z  90  28.4 //
2025-01-21T11:45:01.000Z  90  22.2 //
2025-01-21T11:50:02.000Z  85  20.8 //
2025-01-21T11:55:00.000Z  65  25.8 //
2025-01-21T12:00:01.000Z  90  23.1 //
2025-01-21T12:05:02.000Z  95  20.7 //
2025-01-21T12:10:00.000Z  70  20.6 //
2025-01-21T12:15:01.000Z  95  26.9 //
2025-01-21T12:20:02.000Z  95  25.6 //
2025-01-21T12:25:00.000Z  90  21.2 //
2025-01-21T12:30:01.000Z  95  20.9 //
2025-01-21T12:35:02.000Z  95  19.9 //
2025-01-21T12:40:00.000Z  95  20.9 //
2025-01-21T12:45:01.000Z 100  21.6 //
2025-01-21T12:50:02.000Z  45  16.5 //
2025-01-21T12:55:00.000Z  65  27.6 //
2025-01-21T13:00:01.000Z  40  17.3 //
2025-01-21T13:05:02.000Z  95  20.5 //
2025-01-21T13:10:00.000Z  95  17.4 //
2025-01-21T13:15:01.000Z  95  22.2 //
2025-01-21T13:20:02.000Z  85  28.8 //
2025-01-21T13:25:00.000Z  60  29.6 //
2025-01-21T13:30:01.000Z  80  22.0 //
2025-01-21T13:35:02.000Z  80  25.1 //
2025-01-21T13:40:00.000Z  90  20.9 //
2025-01-21T13:45:01.000Z  90  26.9 //
2025-01-21T13:50:02.000Z  70  30.5 //
2025-01-21T13:55:00.000Z  60  22.4 //
2025-01-21T14:00:01.000Z  90  21.6 //
2025-01-21T14:05:02.000Z  95  20.4 //
2025-01-21T14:25:00.000Z  45   4.3 //
2025-01-21T14:35:02.000Z  95  29.3 //
2025-01-21T14:40:00.000Z  95  24.8 //
2025-01-21T14:45:01.000Z  75  26.4 //
2025-01-21T14:50:02.000Z  80  24.4 //
2025-01-21T14:55:00.000Z 100  23.8 //
2025-01-21T15:00:01.000Z  95  25.7 //
2025-01-21T15:05:02.000Z  90  22.9 //
2025-01-21T16:25:00.000Z  90  11.8 //
2025-01-21T16:35:02.000Z  95  12.3 //
2025-01-21T16:40:00.000Z  95  13.2 //
2025-01-21T16:45:01.000Z  85   7.5 //
2025-01-21T16:50:02.000Z  95   9.0 //
2025-01-21T16:55:00.000Z  90  11.1 //
2025-01-21T17:00:01.000Z 100  11.2 //
2025-01-21T17:05:02.000Z  80   8.4 //
2025-01-21T17:10:00.000Z  80   5.8 //
2025-01-21T17:15:01.000Z  60   7.4 //
2025-01-21T17:20:02.000Z  90  10.3 //
2025-01-21T17:30:01.000Z  95   5.5 //
2025-01-21T17:35:02.000Z  80   5.4 //
2025-01-21T17:40:00.000Z  80   3.8 //
2025-01-21T17:45:01.000Z  85   8.0 //
2025-01-21T17:50:02.000Z  95   7.8 //
2025-01-21T17:55:00.000Z 100   7.9 //
2025-01-21T18:00:01.000Z  95   5.6 //
2025-01-21T18:05:02.000Z  95   5.5 //
2025-01-21T18:10:00.000Z  85   5.2 //
2025-01-21T18:15:01.000Z  90   4.7 //
2025-01-21T18:20:02.000Z  95   5.2 //
2025-01-21T18:25:00.000Z  95   6.2 //
2025-01-21T18:30:01.000Z  90   4.2 //
2025-01-21T18:35:02.000Z 100   2.5 //
2025-01-21T18:40:00.000Z  85   4.4 //
2025-01-21T18:45:01.000Z  70   1.3 //
2025-01-21T18:50:02.000Z  40   0.7 //
2025-01-21T18:55:00.000Z  90   6.0 //
2025-01-21T19:00:01.000Z  95   3.1 //
2025-01-21T19:05:02.000Z  95   1.9 //
2025-01-21T19:10:00.000Z  95   3.4 //
2025-01-21T19:15:01.000Z  95   3.1 //
2025-01-21T19:20:02.000Z  90   2.7 //
2025-01-21T19:25:00.000Z  95   2.4 //
2025-01-21T19:30:01.000Z  80   2.7 //
2025-01-21T19:35:02.000Z  95   3.4 //
2025-01-21T19:40:00.000Z  95   3.6 //
2025-01-21T19:45:01.000Z  95   3.5 //
2025-01-21T19:50:02.000Z  95   3.9 //
2025-01-21T19:55:00.000Z  95   3.5 //
2025-01-21T20:00:01.000Z  75   3.2 //
2025-01-21T20:05:02.000Z  95   2.4 //
2025-01-21T20:10:00.000Z  95   3.1 //
2025-01-21T20:15:01.000Z 100   3.7 //
2025-01-21T20:20:02.000Z  95   2.5 //
2025-01-21T20:25:00.000Z  95   2.2 //
2025-01-21T20:30:01.000Z  95   3.3 //
2025-01-21T20:35:02.000Z  95   2.1 //
2025-01-21T20:40:00.000Z  95   2.6 //
2025-01-21T20:45:01.000Z  90   3.6 //
2025-01-21T20:50:02.000Z  90   3.2 //
2025-01-21T20:55:00.000Z  90   2.5 //
2025-01-21T21:00:01.000Z  95   2.6 //
2025-01-21T21:05:02.000Z  95   2.3 //
2025-01-21T21:10:02.000Z  95   2.3 //
2025-01-21T21:15:01.000Z  95   2.2 //
2025-01-21T21:20:02.000Z 100   2.2 //
2025-01-21T21:25:02.000Z  95   2.2 //
2025-01-21T21:30:01.000Z  95   2.5 //
2025-01-21T21:35:02.000Z  90   3.2 //
2025-01-21T21:40:02.000Z  95   2.0 //
2025-01-21T21:45:01.000Z  95   3.2 //
2025-01-21T21:50:02.000Z 100   2.5 //
2025-01-21T21:55:02.000Z  90   3.3 //
2025-01-21T22:00:01.000Z  90   5.4 //
2025-01-21T22:05:02.000Z  95   4.5 //
2025-01-21T22:10:02.000Z  95   3.6 //
2025-01-21T22:15:01.000Z  95   3.1 //
2025-01-21T22:20:02.000Z  75   2.1 //
2025-01-21T22:25:02.000Z  95   2.7 //
2025-01-21T22:30:01.000Z 100   3.6 //
2025-01-21T22:35:02.000Z  95   3.7 //
2025-01-21T22:40:02.000Z  95   4.0 //
2025-01-21T22:45:01.000Z  95   3.0 //
2025-01-21T22:50:02.000Z  95   4.6 //
2025-01-21T22:55:02.000Z 100   3.3 //
2025-01-21T23:00:01.000Z 100   3.2 //
2025-01-21T23:05:02.000Z 100   3.6 //
2025-01-21T23:10:02.000Z  95   3.1 //
2025-01-21T23:15:01.000Z 100   3.5 //
2025-01-21T23:20:02.000Z 100   2.9 //
2025-01-21T23:25:02.000Z 100   4.0 //
2025-01-21T23:30:01.000Z 100   3.7 //
2025-01-21T23:35:02.000Z  95   3.3 //
2025-01-21T23:40:02.000Z  95   3.3 //
2025-01-21T23:45:01.000Z  95   3.5 //
2025-01-21T23:50:02.000Z 100   3.1 //
2025-01-21T23:55:02.000Z  95   2.5 //
2025-01-22T00:00:01.000Z 100   4.1 //
2025-01-22T00:05:02.000Z  95   4.3 //
2025-01-22T00:10:02.000Z  95   4.1 //
2025-01-22T00:15:01.000Z 100   3.3 //
2025-01-22T00:20:02.000Z 100   3.9 //
2025-01-22T00:25:02.000Z  95   4.4 //
2025-01-22T00:30:01.000Z  90   4.1 //
2025-01-22T00:35:02.000Z  90   2.0 //
2025-01-22T00:40:02.000Z  90   3.0 //
2025-01-22T00:45:01.000Z  90   1.8 //
2025-01-22T00:50:02.000Z  95   4.8 //
2025-01-22T00:55:02.000Z 100   3.6 //
2025-01-22T01:00:01.000Z  85   3.4 //
2025-01-22T01:05:02.000Z  95   3.9 //
2025-01-22T01:10:02.000Z 100   4.5 //
2025-01-22T01:15:01.000Z  95   5.6 //
2025-01-22T01:20:02.000Z  95   3.8 //
2025-01-22T01:25:02.000Z  90   2.9 //
2025-01-22T01:30:01.000Z 100   3.3 //
2025-01-22T01:35:02.000Z  95   2.8 //
2025-01-22T01:40:02.000Z  95   2.3 //
2025-01-22T01:45:01.000Z  95   3.5 //
2025-01-22T01:50:02.000Z  95   3.8 //
2025-01-22T01:55:02.000Z  95   3.5 //
2025-01-22T02:00:01.000Z  90   3.6 //
2025-01-22T02:05:02.000Z  95   4.5 //
2025-01-22T02:10:02.000Z 100   4.0 //
2025-01-22T02:15:01.000Z  95   3.5 //
2025-01-22T02:20:02.000Z  95   3.3 //
2025-01-22T02:25:02.000Z  95   2.5 //
2025-01-22T02:30:01.000Z  90   3.5 //
2025-01-22T02:35:02.000Z  90   3.1 //
2025-01-22T02:40:02.000Z  95   3.1 //
2025-01-22T02:45:01.000Z  95   4.2 //
2025-01-22T02:50:02.000Z  90   4.1 //
2025-01-22T02:55:02.000Z  90   3.0 //
2025-01-22T03:00:01.000Z  90   3.7 //
2025-01-22T03:05:02.000Z  60   3.7 //
2025-01-22T03:10:02.000Z  95   2.9 //
2025-01-22T03:15:01.000Z  90   1.6 //
2025-01-22T03:20:02.000Z  90   2.6 //
2025-01-22T03:25:02.000Z  95   2.3 //
2025-01-22T03:30:01.000Z  90   3.2 //
2025-01-22T03:35:02.000Z  90   3.2 //
2025-01-22T03:40:02.000Z  90   2.9 //
2025-01-22T03:45:01.000Z  90   2.1 //
2025-01-22T03:50:02.000Z  90   2.4 //
2025-01-22T03:55:02.000Z  90   2.0 //
2025-01-22T04:00:01.000Z  90   2.6 //
2025-01-22T04:05:02.000Z  80   1.9 //
2025-01-22T04:10:02.000Z  80   1.1 //
2025-01-22T04:15:01.000Z  45   0.9 //
2025-01-22T04:20:02.000Z  90   2.2 //
2025-01-22T04:25:02.000Z  80   1.9 //
2025-01-22T04:30:01.000Z  80   2.2 //
2025-01-22T04:35:02.000Z  90   1.8 //
2025-01-22T04:40:02.000Z  90   1.8 //
2025-01-22T04:45:01.000Z  80   1.9 //
2025-01-22T04:50:02.000Z  80   1.3 //
2025-01-22T04:55:02.000Z  80   2.0 //
2025-01-22T05:00:01.000Z  80   1.5 //
2025-01-22T05:05:02.000Z  70   2.1 //
2025-01-22T05:10:02.000Z  80   2.1 //
2025-01-22T05:15:01.000Z  60   1.3 //
2025-01-22T05:20:02.000Z  80   1.4 //
2025-01-22T05:25:02.000Z  85   1.1 //
2025-01-22T05:30:01.000Z  80   1.6 //
2025-01-22T05:35:08.000Z  90   1.6 //
2025-01-22T05:40:00.000Z  80   1.4 //
2025-01-22T05:45:01.000Z  80   1.7 //
2025-01-22T05:50:08.000Z  80   1.8 //
2025-01-22T05:55:00.000Z  80   1.5 //
2025-01-22T06:00:01.000Z  80   0.9 //
2025-01-22T06:05:08.000Z  80   1.1 //
2025-01-22T06:10:00.000Z  80   1.8 //
2025-01-22T06:15:01.000Z  85   2.0 //
2025-01-22T06:20:08.000Z  95   1.5 //
2025-01-22T06:25:00.000Z  90   2.0 //
2025-01-22T06:30:01.000Z  85   3.1 //
2025-01-22T06:35:02.000Z  40   1.2 //
2025-01-22T06:40:00.000Z  80   1.7 //
2025-01-22T06:45:01.000Z  95   4.8 //
2025-01-22T06:50:02.000Z  95   4.4 //
2025-01-22T06:55:00.000Z  95   5.0 //
2025-01-22T07:00:01.000Z  95   5.6 //
2025-01-22T07:05:02.000Z  90   6.3 //
2025-01-22T07:10:00.000Z  95   7.3 //
2025-01-22T07:15:01.000Z  90   8.3 //
2025-01-22T07:20:02.000Z  95   9.7 //
2025-01-22T07:25:00.000Z  95   7.3 //
2025-01-22T07:35:02.000Z  95  11.1 //
2025-01-22T07:40:00.000Z  95  10.4 //
2025-01-22T07:45:01.000Z 100   9.7 //
2025-01-22T07:50:02.000Z  95   9.2 //
2025-01-22T08:00:01.000Z  95  10.9 //
2025-01-22T08:05:02.000Z 100  11.2 //
2025-01-22T08:10:00.000Z  95  10.6 //
2025-01-22T08:15:01.000Z  90  14.6 //
2025-01-22T08:20:02.000Z  90  17.2 //
2025-01-22T08:25:00.000Z  90  18.0 //
2025-01-22T08:30:01.000Z  95  13.6 //
2025-01-22T08:35:02.000Z  95  16.1 //
2025-01-22T08:40:00.000Z  95  18.8 //
2025-01-22T08:45:01.000Z 100  21.9 //
2025-01-22T08:50:02.000Z  90  25.4 //
2025-01-22T08:55:00.000Z  80  21.2 //
2025-01-22T09:00:01.000Z 100  23.6 //
2025-01-22T09:05:02.000Z  70  23.2 //
2025-01-22T09:10:00.000Z  60  27.6 //
2025-01-22T09:15:01.000Z  95  24.1 //
2025-01-22T09:20:02.000Z  95  22.0 //
2025-01-22T09:25:00.000Z  90  20.0 //
2025-01-22T09:30:01.000Z  80  21.7 //
2025-01-22T09:35:02.000Z  80  18.6 //
2025-01-22T09:40:00.000Z  95  15.9 //
2025-01-22T09:55:00.000Z  80  28.8 //
2025-01-22T10:00:01.000Z  90  19.8 //
2025-01-22T10:05:02.000Z  75  21.8 //
2025-01-22T10:10:00.000Z  90  22.9 //
2025-01-22T10:15:01.000Z  90  22.4 //
2025-01-22T10:20:02.000Z  65  16.3 //
2025-01-22T10:25:00.000Z  70  19.7 //
2025-01-22T10:30:01.000Z  40  25.2 //
2025-01-22T10:35:02.000Z  95  25.2 //
2025-01-22T10:40:00.000Z  80  28.1 //
2025-01-22T10:45:01.000Z  95  23.5 //
2025-01-22T10:50:02.000Z  85  22.7 //
2025-01-22T10:55:00.000Z  85  25.0 //
2025-01-22T11:00:01.000Z  75  25.2 //
2025-01-22T11:05:02.000Z  75  25.5 //
2025-01-22T11:10:00.000Z  95  28.6 //
2025-01-22T11:15:01.000Z  80  29.6 //
2025-01-22T11:20:02.000Z  95  30.6 //
2025-01-22T11:25:00.000Z  65  23.1 //
2025-01-22T11:30:01.000Z  95  24.2 //
2025-01-22T11:35:02.000Z  85  24.8 //
2025-01-22T11:40:00.000Z  85  24.1 //
2025-01-22T11:45:01.000Z  90  28.2 //
2025-01-22T11:50:02.000Z  90  27.5 //
2025-01-22T11:55:00.000Z  60  32.7 //
"""

# Use StringIO to read the string data as if it were a file
df = pd.read_csv(StringIO(dourbes), delim_whitespace=True, comment="#", header=0, names=["Time", "CS", "TEC", "QD"])

# Convert the time column to datetime
df['Time'] = pd.to_datetime(df['Time'])

plt.plot(df['Time'], df['TEC'], marker='o', label='TEC [10^16 m^-2]')

# Final plotting setup
plt.xlabel('Time')
plt.ylabel('STEC (TECU)')
plt.title('STEC vs Time for each PRN')
plt.legend(loc="lower left", ncol=len(prn_i)//4)
plt.grid()
plt.ylim(-20, 50)
#plt.tight_layout()
plt.show()
