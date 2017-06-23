# Scripts for modeling the value of vesting stock options

These scripts provide the Python source code for the model in [this blog post](http://benkuhn.net/optopt), which compares the expected value of a job offer from a startup (with stock options and lower cash compensation) to that from a big company.

If you want to use this to look at your own offer, here's where to go:

1. Make a python 3 virtualenv.
1. `pip install -r requirements.txt`
1. Open `options.py`. Edit the `COMMON_PARAMS` dict with the parameters from your offer. Edit the calls to `sensitivity_analysis` at the end to control how the parameter space is explored in the sensitivity analysis.
1. Run `python options.py` to get the expected value of the offer and produce a sensitivity analysis.
1. Run `python plot_trajectory.py` to produce the equivalent of that post's first plot (heatmap trajectory of valuations, when you should quit, and probability of staying).
1. Run `python plot_sensitivity.py` to produce the equivalent of that post's second and third plots (analysis of sensitivity to various parameters).
