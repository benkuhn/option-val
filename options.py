from collections import namedtuple
from decimal import Decimal as D
import numpy as np
import scipy as sp
import datetime

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

    @property
    def ts_volatility(self):
        return self.annual_volatility / np.sqrt(self.annual_timesteps)

    @property
    def ts_growth(self):
        return self.annual_growth ** (1/self.annual_timesteps)

    @property
    def ts_vesting_interval(self):
        return self.vesting_period * self.annual_timesteps

    @property
    def ts_horizon(self):
        return self.horizon_years * self.annual_timesteps

    @property
    def ts_gain(self):
        """Amount the valuation goes up per one 'up' timestep"""
        # Source: http://www.maths.usyd.edu.au/u/UG/SM/MATH3075/r/Slides_7_Binomial_Market_Model.pdf
        return np.exp(self.ts_volatility)

    @property
    def ts_loss(self):
        return 1/self.ts_gain

    @property
    def ts_opportunity_cost(self):
        return self.opportunity_cost / self.annual_timesteps

    @property
    def p_growth(self):
        """Probability that each timestep is an 'up' timestep"""
        # Source: http://www.maths.usyd.edu.au/u/UG/SM/MATH3075/r/Slides_7_Binomial_Market_Model.pdf
        return (self.ts_growth - self.ts_loss) / (self.ts_gain - self.ts_loss)

    @property
    def ts_vesting_increments(self):
        cliff = self.annual_timesteps
        end = self.ts_vesting_interval
        month_len = self.annual_timesteps // 12
        return [0] + list(range(cliff, end + 1, month_len))

    @property
    def p_loss(self):
        return 1 - self.p_growth

    def get_valuation(self, i):
        """Return the valuation of the company at node i"""
        return self.initial_valuation * self.ts_gain ** (i.n_up - i.n_down)


def memoize(fn):
    """Memoize a method of the Params subclass, for dynamic programming"""
    # inval the cache bc we're probably redefining a function
    def inner(self, *args):
        if args not in self.cache:
            self.cache[args] = fn(self, *args)
        return self.cache[args]
    return inner


class FixedHorizonModel(ModelBase):

    @memoize
    def get_payoff(self, i, t_quit=None, t_vesting_end=None):
        """Get the expected payoff from state i, if you quit at time t_quit"""
        assert (t_quit is not None) == (t_vesting_end is not None)
        if i.t == self.ts_vesting_interval and t_quit is None:
            # The trade you got in your offer is over now, so pretend you
            # stopped working. In reality, you'll probably be offered more
            # trades at this point (via refresher grants) so this
            # underestimates the value of the initial trade.
            return self.get_payoff(i, i.t, i.t)
        elif i.t == self.ts_horizon:
            # TODO(ben): apply time discounting the opportunity cost
            cost = t_quit * self.ts_opportunity_cost
            vested_fraction = t_vesting_end / self.ts_vesting_interval

            # extrapolate the payoff to the end of the horizon
            full_payoff = max(self.get_valuation(i) - self.strike_price, 0)
            return full_payoff * self.ownership_fraction * vested_fraction - cost
        elif t_quit is not None:
            # we already quit, so just run through to the end
            return (self.p_growth * self.get_payoff(i.go_up(), t_quit, t_vesting_end)
                    + self.p_loss * self.get_payoff(i.go_down(), t_quit, t_vesting_end))
        else:
            return max(self.get_payoff_if_quit(i), self.get_payoff_if_stay(i))

    def get_payoff_if_quit(self, i):
        """Get the expected payoff from state i, if you quit exactly then"""
        # vesting ends at the last "vesting increment"
        t_vesting_end = max(t for t in self.ts_vesting_increments if t <= i.t)
        return self.get_payoff(i, i.t, t_vesting_end)

    def get_payoff_if_stay(self, i):
        """Get the expected payoff from state i, if you quit exactly then"""
        return (self.p_growth * self.get_payoff(i.go_up())
                + self.p_loss * self.get_payoff(i.go_down()))


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
    initial_valuation=2e7,
    strike_price=5e6,
    annual_volatility=1.0,
    annual_timesteps=24,
    ownership_fraction=0.006,
    opportunity_cost=60000,
)

# params = FixedHorizonParams(
#     **COMMON_PARAMS,
#     horizon_years=7
# )

params = NaiveModel(
    **COMMON_PARAMS,
    horizon_years=7
)

print(params.get_payoff(I(0,0)))
