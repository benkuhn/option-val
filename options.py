from collections import namedtuple
from decimal import Decimal as D
import datetime
import math
import csv

class I(namedtuple('IndexBase', 't n_up')):
    """Index into the lattice of option states.

    fields:
        t: total timesteps elapsed
        n_up: number of timesteps the valuation went up
    """
    def __init__(self, t, n_up):
        assert 0 <= self.n_up <= self.t, \
            "We want 0 <= %s <= %s" % (self.n_up, self.t)

    @property
    def n_down(self):
        return self.t - self.n_up

    def go_up(self):
        return I(self.t + 1, self.n_up + 1)

    def go_down(self):
        return I(self.t + 1, self.n_up)


def memoize(fn):
    """Memoize a method of the Params subclass, for dynamic programming"""
    # inval the cache bc we're probably redefining a function
    def inner(self, *args):
        cache = self.cache.setdefault(fn.__name__, {})
        if args not in cache:
            cache[args] = fn(self, *args)
        return cache[args]
    return inner


class ModelBase:
    # company parameters
    initial_valuation = None
    strike_price = None
    annual_volatility = 0.54  # 2x S&P 500 annual vol
    annual_growth = 1.08  # long run growth rate of equities
    # numerics parameters
    annual_timesteps = 12
    # offer parameters
    vesting_period = 4  # years
    ownership_fraction = None
    opportunity_cost = None  # annual $ lost from not working at bigco
    horizon_years = 7


    def __init__(self, **kwargs):
        self.cache = {}
        for k, v in kwargs.items():
            assert hasattr(self, k)
            setattr(self, k, v)

        self.ts_volatility = self.annual_volatility / math.sqrt(self.annual_timesteps)
        # Amount the valuation goes up per one 'up' timestep
        # Source: http://www.maths.usyd.edu.au/u/UG/SM/MATH3075/r/Slides_7_Binomial_Market_Model.pdf
        self.ts_gain = math.exp(self.ts_volatility)
        self.ts_growth = self.annual_growth ** (1/self.annual_timesteps)
        self.ts_vesting_interval = self.vesting_period * self.annual_timesteps
        self.ts_horizon = int(self.horizon_years * self.annual_timesteps)
        self.ts_loss = 1/self.ts_gain
        self.ts_opportunity_cost = self.opportunity_cost / self.annual_timesteps
        # Probability that each timestep is an 'up' timestep
        # Source: http://www.maths.usyd.edu.au/u/UG/SM/MATH3075/r/Slides_7_Binomial_Market_Model.pdf
        self.p_growth = (self.ts_growth - self.ts_loss) / (self.ts_gain - self.ts_loss)
        self.p_loss = 1 - self.p_growth

        # vesting increments
        cliff = self.annual_timesteps
        end = self.ts_vesting_interval
        month_len = max(1, self.annual_timesteps // 12)
        self.ts_vesting_increments = [0] + list(range(cliff, end + 1, month_len))

    def get_valuation(self, i):
        """Return the valuation of the company at node i"""
        return self.initial_valuation * self.ts_gain ** (i.n_up - i.n_down)


class FixedHorizonModel(ModelBase):

    @memoize
    def get_payoff(self, i, t_quit=None):
        """Get the expected payoff from state i, if you quit exactly then"""
        if t_quit is None:
            # We haven't quit yet. Should we quit now?
            payoff_if_quit = self.get_payoff(i, i.t)
            if i.t == self.ts_vesting_interval:
                # The trade you got in your offer is over now, so pretend you
                # stopped working. In reality, you'll probably be offered more
                # trades at this point (via refresher grants) so this
                # underestimates the value of the initial trade.
                return payoff_if_quit
            payoff_if_stay = (self.p_growth * self.get_payoff(i.go_up())
                              + self.p_loss * self.get_payoff(i.go_down()))
            return max(payoff_if_stay, payoff_if_quit)
        elif i.t == self.ts_horizon:
            # TODO(ben): apply time discounting the opportunity cost
            cost = t_quit * self.ts_opportunity_cost
            # vesting ends at the last "vesting increment"
            t_vesting_end = max(t for t in self.ts_vesting_increments if t <= t_quit)
            vested_fraction = t_vesting_end / self.ts_vesting_interval

            # exercise iff valuation > strike at time horizon
            full_payoff = max(self.get_valuation(i) - self.strike_price, 0)
            return full_payoff * self.ownership_fraction * vested_fraction - cost
        else:
            # we already quit so just run through the end
            return (self.p_growth * self.get_payoff(i.go_up(), t_quit)
                    + self.p_loss * self.get_payoff(i.go_down(), t_quit))
        return self.get_payoff(i, i.t, t_vesting_end)


class NaiveModel(ModelBase):
    @memoize
    def get_payoff(self, i):
        """Get the expected payoff from state i if you can't quit"""
        if i.t == self.ts_horizon:
            cost = self.vesting_period * self.opportunity_cost
            payoff = max(self.get_valuation(i) - self.strike_price, 0)
            return payoff * self.ownership_fraction - cost
        else:
            return (self.p_growth * self.get_payoff(i.go_up())
                    + self.p_loss * self.get_payoff(i.go_down()))


COMMON_PARAMS = dict(
    initial_valuation=1e7,
    strike_price=2.5e6,
    # empirically, the difference between the values with 12 and 24 timestamps
    # is a few percentage points and 12 is way faster than 24
    annual_timesteps=12,
    ownership_fraction=0.01,
    opportunity_cost=50000,
    annual_volatility=1.0,
    horizon_years=7,
)

def sensitivity_analysis(param_name, values):
    print('Analyzing sensitivity to', param_name)
    ROW_TEMPLATE = '% 20s% 15s% 15s'
    with open(param_name + '.csv', 'w') as out:
        w = csv.DictWriter(out, [param_name, 'naive', 'fixed_horizon'])
        w.writeheader()
        print(ROW_TEMPLATE % (param_name, 'naive', 'fixed_horizon'))
        for value in values:
            params = dict(COMMON_PARAMS)
            params[param_name] = value
            naive_val = NaiveModel(**params).get_payoff(I(0, 0))
            fh_val = FixedHorizonModel(**params).get_payoff(I(0, 0))
            print(ROW_TEMPLATE % (value, int(naive_val), int(fh_val)))
            w.writerow({
                param_name: value,
                'naive': naive_val,
                'fixed_horizon': fh_val,
            })

if __name__ == '__main__':
    # sensitivity_analysis('horizon_years', range(4, 12))
    sensitivity_analysis('opportunity_cost', range(10000, 160000, 10000))
    sensitivity_analysis('strike_price', [5e5, 1e6, 2.5e6, 5e6, 1e7, 2.5e7, 5e7, 1e8])
    sensitivity_analysis('annual_growth', [1 + x/100 for x in range(0, 16, 2)])
    sensitivity_analysis('annual_volatility', [x/4 for x in range(1, 9)])

    # params = FixedHorizonParams(
    #     **COMMON_PARAMS,
    # )

    # params = FixedHorizonModel(
    #     **COMMON_PARAMS,
    # )

    # print(params.get_payoff(I(0,0)))

    # print(NaiveModel(**COMMON_PARAMS).get_payoff(I(0,0)))
