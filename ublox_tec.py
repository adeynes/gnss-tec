import json
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from matplotlib import cm
from matplotlib.colors import ListedColormap
import math

file_path = "ubloxdata_20250312"

SMOOTH = 3600

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 24

plt.rc('font', **{'family': 'serif', 'size': SMALL_SIZE})
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def load_data(file_path):
    with open(file_path, 'r') as file:
        raw_data = file.read()
    # Split the concatenated JSON blocks
    json_blocks = raw_data.replace('}{', '}\n{').split('\n')
    return [json.loads(block) for block in json_blocks]

data_blocks = load_data(file_path)
print(len(data_blocks))

print(data_blocks[-1]["tow"])

tecps = {}
for block in data_blocks:
    for prn, tec in block["tecps"].items():
        if int(prn) not in tecps:
            tecps[int(prn)] = {"tows": [], "tecps": []}

        tow = round(block["tow"])
        if tow > 309000:
            date_str = "20250309T00:00:00"
        else:
            date_str = "20250316T00:00:00"
        tecps[int(prn)]["tows"].append(np.datetime64(datetime.fromisoformat(date_str) + timedelta(seconds=tow)))
        tecps[int(prn)]["tecps"].append(tec)

def smooth_data(values, window_size=60):
    if len(values) < window_size:
        window_size = len(values)
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid')


def remove_large_slopes(t, stec, max_slope):
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
    t_sec = np.array([(x - t[0]).item().total_seconds() for x in t])
    stec = np.array(stec, dtype=float)
    
    # Calculate slopes
    slopes = np.abs(np.diff(stec) / np.diff(t_sec))
    
    # Create a mask for points where the slope is within the allowed range
    mask = slopes <= max_slope
    
    # Include the last valid point by appending True to the mask
    valid_indices = np.hstack(([True], mask)) & np.hstack((mask, [True]))
    
    # Filter x and y using the mask
    filtered_t = [t[i] for i in range(len(t)) if valid_indices[i]]
    filtered_stec = [stec[i] for i in range(len(stec)) if valid_indices[i]]
    
    return filtered_t, filtered_stec

def add_gaps(x, y, max_gap=timedelta(minutes=2)):
    """
    Inserts NaT values into datetime x and NaN into y where gaps in x exceed max_gap.
    """
    x = np.array(x, dtype='datetime64[us]')  # Ensure x is numpy datetime64
    y = np.array(y, dtype=float)  # Ensure y is a float array
    
    # Find gaps in x
    gaps = np.diff(x) > max_gap
    if not np.any(gaps):
        return x, y  # No gaps to handle
    
    # Split and insert NaT where gaps occur
    new_x = []
    new_y = []
    for i in range(len(x) - 1):
        new_x.append(x[i])
        new_y.append(y[i])
        if gaps[i]:  # Add NaT and NaN if there's a gap
            new_x.append(np.datetime64('NaT'))
            new_y.append(np.nan)
    # Append the last point
    new_x.append(x[-1])
    new_y.append(y[-1])
    
    return np.array(new_x), np.array(new_y)


avg = {}
tecs_ublox = {}

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
num_prns = len(tecps)
colors = [colormap((i % num_colors) / num_colors) for i in range(num_prns)]  # Generate unique colors for each PRN
line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
j = 0
ublox_stecs = {}

for svId, svTecps in dict(sorted(tecps.items())).items():
    stecs = smooth_data(svTecps["tecps"], SMOOTH)
    ts = svTecps["tows"][:len(stecs)]

    ts, stecs = remove_large_slopes(ts, stecs, 1/50)

    ts, stecs = add_gaps(ts, stecs)

    np.datetime_as_string([(np.round(t.astype('datetime64[s]').astype('int64') / 60) * 60).astype('datetime64[s]').astype('datetime64[m]') for t in ts]).tolist()
    ublox_stecs[svId] = {"tows": np.datetime_as_string([(np.round(t.astype('datetime64[s]').astype('int64') / 60) * 60).astype('datetime64[s]').astype('datetime64[m]') for t in ts]).tolist(), "stecs": stecs.tolist()}

    for t, stec in zip(ts, stecs):
        if t not in avg:
            avg[t] = []
        avg[t].append(stec)

    plt.plot(ts, stecs, linewidth=0.8, alpha=0.5, color="gray")
    j += 1

for t, stecs in avg.items():
	avg[t] = sum(stecs)/len(stecs)

perm = np.argsort(list(avg.keys()))

t = list(avg.keys())
avgtecs = list(avg.values())

#avgtecs_ublox = smooth_data([avgtecs[i] * 2.4 / 3.2 + 65/3.2 for i in perm], SMOOTH)
avgtecs_ublox = smooth_data([avgtecs[i] for i in perm], SMOOTH)
t_ublox = [t[perm[i]] for i in range(len(avgtecs_ublox))]


plt.plot(t_ublox, avgtecs_ublox, linewidth=3, label="Bifréquence", color="limegreen")
plt.grid()
plt.xlabel('Date', labelpad=15)
plt.ylabel('STEC (TECU)', labelpad=15)
plt.title("STEC en fonction du temps (ublox, bifréquence)", fontsize=BIGGER_SIZE, pad=30)
plt.ylim(-80, 100)
#plt.legend(loc="upper left", ncol=9)
plt.show()

