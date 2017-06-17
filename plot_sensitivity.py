import matplotlib.pyplot as plt
import pandas as pd
import seaborn

def plot_series(name, xlabel=None):
    data = pd.read_csv('%s.csv' % name)
    plt.plot(data[name], data.fixed_horizon, label='optimal quitting')
    plt.plot(data[name], data.naive, label='no quitting')
    xlabel = xlabel or name.replace('_', ' ')
    plt.xlabel(xlabel)
    plt.title('sensitivity to ' + xlabel.split('(')[0])
    plt.ylabel('net offer value ($)')
    plt.legend()
    plt.axhline(0, color='gray')

### Sensitivity to opportunity cost

plt.figure(figsize=(8, 4))

plot_series('opportunity_cost', xlabel='opportunity cost (annual, $)')
plt.tight_layout()
plt.savefig('sensitivity_opp_cost_2x', dpi=160)
plt.savefig('sensitivity_opp_cost', dpi=80)

### Sensitivity to others

plt.figure(figsize=(8, 8))

PV = 3
PH = 2

ax1 = plt.subplot(PV, PH, 1)
## annual volatility
plot_series('annual_volatility', xlabel='volatility (%, annualized)')

## growth period
ax = plt.subplot(PV, PH, 2, sharey=ax1)
plt.setp(ax.get_yticklabels(), visible=False)
plot_series('horizon_years', xlabel='growth period (years)')

## growth rate
ax = plt.subplot(PV, PH, 3, sharey=ax1)
plot_series('annual_growth', xlabel='growth rate')

## strike price
ax = plt.subplot(PV, PH, 4, sharey=ax1)
plt.setp(ax.get_yticklabels(), visible=False)
plot_series('strike_price', xlabel='strike price (x $10m)')

## growth rate
ax = plt.subplot(PV, PH, 5, sharey=ax1)
plot_series('annual_discount_rate', xlabel='discount rate')

plt.tight_layout()
plt.savefig('sensitivity_others_2x', dpi=160)
plt.savefig('sensitivity_others', dpi=80)
