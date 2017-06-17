import math

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec, ticker
import seaborn

import options

params = dict(options.COMMON_PARAMS)
params['annual_timesteps'] = 36
model = options.FixedHorizonModel(**params)

## get the data
indices = [options.I(t, n_up)
           for t in range(model.ts_horizon+1)
           for n_up in range(t+1)]
df = pd.DataFrame(columns=['t', 'n_up', 'p', 'val', 'quit', 'p_quit_by_now'],
                  index=range(len(indices)))
for n, i in enumerate(indices):
    should_quit = (i.t > model.ts_vesting_interval
                   or model.get_payoff(i, i.t) >= model.get_payoff(i))
    df.loc[n] = (i.t, i.n_up, model.get_p_n_up(i), model.get_valuation(i),
                 should_quit, model.p_quit_before_or_at(i))

# prob of quitting at timestep t
p_quit = pd.Series(index=range(model.ts_horizon+1))
for n in p_quit.index:
    df_for_n = df[df.t == n]
    p_quit[n] = (df_for_n.p * df_for_n.p_quit_by_now).sum()

# valuation above which you should stay at timestep t
stay_valuation = pd.Series(index=p_quit.index)
for t in range(model.ts_vesting_interval):
    if model.get_payoff(options.I(t, 0), t) < model.get_payoff(options.I(t, 0)):
        # should never quit at this timestep
        continue
    for n_up in range(1, t+1):
        i = options.I(t, n_up)
        if model.get_payoff(i, t) < model.get_payoff(i):
            stay_valuation[t] = model.get_valuation(i)
            break

# Normalize the heatmap probabilities + remove valuations < $10 and > $100b
LOG_Y_MIN = 1
LOG_Y_MAX = 11
df_clean = df[(df.val > 10 ** LOG_Y_MIN) & (df.val < 10 ** LOG_Y_MAX)].copy()
for t in range(model.ts_horizon+1):
    ps = df_clean.p[df_clean.t == t]
    pmax = ps.max()
    for i in ps.index:
        df_clean.p[i] = ps[i] / pmax

# Construct the heatmap grid
ASPECT_RATIO = 1
GRID_NX = max(df.t)
GRID_NY = int(GRID_NX * ASPECT_RATIO)
grid = np.zeros([GRID_NX, GRID_NY])
log_y_points = np.linspace(1, 11, GRID_NY + 1)
log_y_midpoints = (log_y_points[1:] + log_y_points[:-1])/2
for t in range(1, GRID_NX + 1):
    vals = df_clean[df_clean.t == t]
    f = scipy.interpolate.interp1d(vals.val, vals.p, fill_value=0, bounds_error=False)
    q = scipy.interpolate.interp1d(vals.val, vals.quit, fill_value=(0, 1),
                                   bounds_error=False, kind='nearest')
    for i, log_y in enumerate(log_y_midpoints):
        y = math.pow(10, log_y)
        quit = q(y)
        sign = -1 if quit else 1
        grid[GRID_NY-i-1, t-1] = sign * f(y)

# Plot the figure
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

ax1 = plt.subplot(gs[0])
ax1.set_ylim(0, GRID_NY)
ticks_per_increment = GRID_NY // (LOG_Y_MAX - LOG_Y_MIN)
ylabels = list(reversed(['1e' + str(x//ticks_per_increment + 1)
           if x % ticks_per_increment == 0 else ''
           for x in range(GRID_NY)]))
seaborn.heatmap(grid, yticklabels=ylabels, xticklabels=False, cbar=False, square=True)
plt.ylabel('valuation ($)')
plt.title('Valuation and employment trajectory')
pal = seaborn.color_palette("RdBu_r", 10)
stay = mpatches.Patch(color=pal[-1], label='should stay')
leave = mpatches.Patch(color=pal[0], label='should leave')
plt.legend(handles=[stay, leave])

ax = plt.subplot(gs[1], sharex=ax1)
ax.set_xlim(0, GRID_NX)
ax.xaxis.set_major_locator(ticker.MultipleLocator(model.annual_timesteps))
ax.set_xticklabels(range(-1, 8))
plt.xlabel('time (years)')
plt.ylabel('probability of staying')
ax.set_ylim(-0.01, 1.01)
plt.plot(p_quit.index, 1 - p_quit)

plt.tight_layout()
plt.savefig('trajectory_2x', dpi=160)
plt.savefig('trajectory', dpi=80)
