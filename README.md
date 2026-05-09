# UliEngineering

[![Unit tests](https://github.com/ulikoehler/UliEngineering/actions/workflows/test.yml/badge.svg)](https://github.com/ulikoehler/UliEngineering/actions/workflows/test.yml) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/f59d862b25524914b29ec5f0c0b80e7f)](https://www.codacy.com/gh/ulikoehler/UliEngineering/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ulikoehler/UliEngineering&amp;utm_campaign=Badge_Grade) [![codecov](https://codecov.io/gh/ulikoehler/UliEngineering/branch/master/graph/badge.svg?token=qnmVG2tYQq)](https://codecov.io/gh/ulikoehler/UliEngineering) [![Code Climate](https://codeclimate.com/github/ulikoehler/UliEngineering/badges/gpa.svg)](https://codeclimate.com/github/ulikoehler/UliEngineering) ![License](https://img.shields.io/github/license/ulikoehler/UliEngineering.svg) 

A Python3 library for:
    - Data science
    - Electronics Engineering
    - Specialized algorithms
    - Physics
which contains a collection of functions I haven't found elsewhere.
Some algorithms have also been accepted on my blog [https://techoverflow.net/](https://techoverflow.net/)

Pull requests and bugreports of any kind are happily accepted.

## Installation

Run this command on your favourite shell:

```sh
sudo pip3 install UliEngineering
```

In order to install the latest bleeding-edge version, use:

```sh
sudo pip3 install git+https://github.com/ulikoehler/UliEngineering.git
```

We *highly recommend* you also install scipy:
```sh
sudo pip3 install scipy
```
(you can also use `sudo apt install python3-scipy` on Debian-based Linux distros).

While *scipy* is technically optional and listed as an optional dependency, many UliEngineering modules can't be used without it, including (might not be complete!):

  * UliEngineering.Length
  * UliEngineering.SignalProcessing.Interpolation
  * UliEngineering.SignalProcessing.Correlation
  * UliEngineering.Physics.JohnsonNyquistNoise
  * Some functions in UliEngineering.SignalProcessing.Simulation
  * Some functions in UliEngineering.SignalProcessing.Selection

After that you can use UliEngineering from any Python3 instance. Example:

```
$ python3
Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from UliEngineering.EngineerIO import *
>>> format_value(0.015, "V")
'15.0 mV'
```

## Getting started

On [my blog](https://techoverflow.net), I've posted several easy-to-use examples on how to solve real-world problems using UliEngineering:
  - [Engineering for the super-lazy: Solving equations without activating your brain](https://techoverflow.net/2015/02/21/engineering-for-the-super-lazy-solving-equations-without-activating-your-brain/)
  - [Finding the nearest E96 resistor value in Python](https://techoverflow.net/2015/05/19/finding-the-nearest-e96-resistor-value-in-python/)
  - [Calculating the NCP380 ILIM resistor using Python](https://techoverflow.net/2015/05/19/calculating-the-ncp380-ilim-resistor-using-python/)
  - [Normalizing electronics engineering value notations using Python](https://techoverflow.net/2015/06/09/normalizing-electronics-engineering-value-notations-using-python/)
  - [Computing the LP2980 adjust resistor using Python](https://techoverflow.net/2015/06/09/computing-the-lp2980-adjust-resistor-using-python/)
  - [Accurate calculation of PT100/PT1000 temperature from resistance](https://techoverflow.net/2016/01/02/accurate-calculation-of-pt100pt1000-temperature-from-resistance/)
  - [Reading a shapefile directly from a zip using pyshp](https://techoverflow.net/2017/02/22/reading-a-shapefile-directly-from-a-zip-using-pyshp/)
  - [Computing bounding box for a list of coordinates in Python](https://techoverflow.net/2017/02/23/computing-bounding-box-for-a-list-of-coordinates-in-python/)
  - [How to solve git fatal: no configured push destination](https://techoverflow.net/2017/08/09/how-to-solve-git-fatal-no-configured-push-destination/)
  - [Easy zero crossing detection in Python using UliEngineering](https://techoverflow.net/2018/12/31/easy-zero-crossing-detection-in-python-using-uliengineering/)
  - [Easily generate sine/cosine wave test data in Python using UliEngineering](https://techoverflow.net/2018/12/31/easily-generate-sine-cosine-waveform-data-in-python-using-uliengineering/)
  - [Easily generate square/triangle/sawtooth/inverse sawtooth waveform data in Python using UliEngineering](https://techoverflow.net/2018/12/31/easily-generate-square-triangle-sawtooth-inverse-sawtooth-waveform-data-in-python-using-uliengineering/)
  - [Easily compute & visualize FFTs in Python using UliEngineering](https://techoverflow.net/2018/12/31/easily-compute-visualize-ffts-in-python-using-uliengineering/)
  - [Computing the temperature under load of your electronics components in Python using UliEngineering](https://techoverflow.net/2019/05/09/computing-the-temperature-under-load-of-your-electronics-components-in-python-using-uliengineering/)
  - [Computing crystal load capacitance using Python & UliEngineering](https://techoverflow.net/2019/05/10/computing-crystal-load-capacitance-using-python-uliengineering/)
  - [How to convert Celsius/Fahrenheit/Kelvin temperatures in Python using UliEngineering](https://techoverflow.net/2019/05/10/how-to-convert-celsius-fahrenheit-kelvin-temperatures-in-python-using-uliengineering/)
  - [How to get current datetime as NumPy datetime (np.datetime64)](https://techoverflow.net/2019/05/12/how-to-get-current-datetime-as-numpy-datetime-np-datetime64/)
  - [How to convert NumPy timedelta (np.timedelta64) object to integer](https://techoverflow.net/2019/05/13/how-to-convert-numpy-timedelta-np-timedelta64-object-to-integer/)
  - [How to get unit/resolution of NumPy np.timedelta64 object](https://techoverflow.net/2019/05/13/how-to-get-unit-resolution-of-numpy-np-timedelta64-object/)
  - [How to get unit/resolution of NumPy np.datetime64 object](https://techoverflow.net/2019/05/13/how-to-get-unit-resolution-of-numpy-np-datetime64-object/)
  - [How to iterate all days of year using Python](https://techoverflow.net/2019/05/16/how-to-iterate-all-days-of-year-using-python/)
  - [How to get number of days in month in Python](https://techoverflow.net/2019/05/16/how-to-get-number-of-days-in-month-in-python/)
  - [Capacitive reactance online calculator (Python code)](https://techoverflow.net/2019/07/30/capacitive-reactance-online-calculator-python-code/)
  - [Inductive reactance online calculator (Python code)](https://techoverflow.net/2019/07/30/inductive-reactance-online-calculator-python-code/)
  - [Capacitor energy from capacitance and voltage online calculator (Python code)](https://techoverflow.net/2019/08/07/capacitor-energy-from-capacitance-and-voltage-online-calculator-python-code/)
  - [How to compute candelas from lumens by apex angle](https://techoverflow.net/2019/08/19/lumen-to-candela-online-calculator-python-code/)
  - [How to create pandas time series dataframe example dataset](https://techoverflow.net/2020/05/25/how-to-create-pandas-time-series-dataframe-example-dataset/)
  - [Matplotlib custom SI prefix unit tick formatter](https://techoverflow.net/2020/05/29/matplotlib-custom-si-prefix-unit-tick-formatter/)
  - [How to skip first element of a generator/iterator in Python](https://techoverflow.net/2020/05/31/how-to-skip-first-element-of-a-generator-iterator-in-python/)
  - [How to compute crystal load capacitors using Python](https://techoverflow.net/2021/07/26/how-to-compute-crystal-load-capacitors-using-python/)
  - [How to tune your crystal oscillator to get the best possible frequency accuracy](https://techoverflow.net/2021/07/26/how-to-tune-your-crystal-oscillator-to-get-the-best-possible-frequency-accuracy/)
  - [How to compute resistor voltage divider ratio using Python](https://techoverflow.net/2021/12/16/how-to-compute-resistor-voltage-divider-ratio-using-python/)
  - [How to fix tox AttributeError: module 'virtualenv.create.via_global_ref.builtin.cpython.mac_os' has no attribute 'cpython2macosarmframework'](https://techoverflow.net/2022/02/03/how-to-fix-tox-attributeerror-module-virtualenv-create-via_global_ref-builtin-cpython-mac_os-has-no-attribute-cpython2macosarmframework/)
  - [How to DC-sweep resistive voltage divider using PySpice](https://techoverflow.net/2022/03/10/how-to-dc-sweep-resistive-voltage-divider-using-pyspice/)
  - [How to simulate resistive voltage divider using PySpice](https://techoverflow.net/2022/03/10/how-to-simulate-resistive-voltage-divider-using-pyspice/)
  - [How to compute & plot sun path diagram using skyfield in Python](https://techoverflow.net/2022/06/19/how-to-compute-plot-sun-path-diagram-using-skyfield-in-python/)
  - [How to generate datetime for every hour on a given day in Python](https://techoverflow.net/2022/06/19/how-to-generate-datetime-for-every-hour-on-a-given-day-in-python/)
  - [How to generate datetime for every minute on a given day in Python](https://techoverflow.net/2022/06/19/how-to-generate-datetime-for-every-minute-on-a-given-day-in-python/)
  - [How to generate datetime for every second on a given day in Python](https://techoverflow.net/2022/06/19/how-to-generate-datetime-for-every-second-on-a-given-day-in-python/)
  - [How to generate filename with date & time in Python](https://techoverflow.net/2022/09/02/how-to-generate-filename-with-date-time-in-python/)
  - [How to compute MRI Larmor frequency for a given magnetic field using Python](https://techoverflow.net/2023/02/04/how-to-compute-mri-larmor-frequency-for-a-given-magnetic-field-using-python/)
  - [How to compute voltage divider output voltage using Python](https://techoverflow.net/2023/02/05/how-to-compute-voltage-divider-output-voltage-using-python/)
  - [How to format axis as dB (decibel) using matplotlib](https://techoverflow.net/2023/03/13/how-to-format-axis-as-db-decibel-using-matplotlib/)
  - [How to compute Buck/Boost/LDO output voltage by feedback resistors using Python](https://techoverflow.net/2023/04/09/how-to-compute-buck-boost-ldo-output-voltage-by-feedback-resistors-using-python/)
  - [How to compute the weight of a titanium or stainless steel rod using UliEngineering in Python](https://techoverflow.net/2023/05/21/how-to-compute-the-weight-of-a-titanium-or-stainless-steel-rod-using-uliengineering-in-python/)
  - [How to ignore warnings in Python unit tests (pytest / tox)](https://techoverflow.net/2023/07/29/how-to-ignore-warnings-in-python-unit-tests-pytest-tox/)
  - [How to compute non-inverting OpAmp amplifier gain using UliEngineering in Python](https://techoverflow.net/2023/09/18/how-to-compute-non-inverting-opamp-amplifier-gain-using-uliengineering-in-python/)
  - [How to compute MOSFET gate charge loss power using Python](https://techoverflow.net/2024/02/11/how-to-compute-mosfet-gate-charge-loss-power-using-python/)
  - [How to compute capacitor constant current charge/discharge time using Python](https://techoverflow.net/2024/06/19/how-to-compute-capacitor-constant-current-discharge-time-using-python/)
  - [Calculating diode maximum power disspation using Python](https://techoverflow.net/2024/08/25/calculating-diode-maximum-power-disspation-using-python/)
  - [Simple buck regulator inductor selection using Python](https://techoverflow.net/2024/08/30/simple-buck-regulator-inductor-selection-using-python/)
  - [How to plot MOSFET Gate capacitance vs gate drive voltage in Python](https://techoverflow.net/2024/09/11/how-to-plot-mosfet-gate-capacitance-vs-gate-drive-voltage-in-python/)
  - [How to select crystal & load capacitors for the DP83T510E 10Base-T1L Single-Pair Ethernet PHY](https://techoverflow.net/2024/09/15/how-to-select-crystal-load-capacitors-for-the-dp83t510e-10base-t1l-single-pair-ethernet-phy/)
  - [How to fix Poetry publish HTTP Error 403: Invalid or non-existent authentication information](https://techoverflow.net/2024/09/23/how-to-fix-poetry-publish-http-error-403-invalid-or-non-existent-authentication-information/)
  - [Advanced LED series resistor value & power disspation calculation using Python](https://techoverflow.net/2024/11/05/advanced-led-series-resistor-value-power-disspation-calculation-using-python/)
  - [How to plot MOSFET with gate series resistor RC lowpass cutoff frequency](https://techoverflow.net/2025/05/27/how-to-plot-mosfet-with-gate-series-resistor-rc-lowpass-cutoff-frequency/)
  - [CAN bus split termination: What are typical component values?](https://techoverflow.net/2025/06/03/can-bus-split-termination-what-are-typical-component-values/)
  - [200nm to 200μm water extinction coefficient model in Python](https://techoverflow.net/2025/06/22/200nm-to-200mm-water-extinction-coefficient-model-in-python/)
  - [How to compute the amount in moles of XX grams of DNA using Python](https://techoverflow.net/2025/06/22/how-to-compute-the-amount-in-moles-of-xx-grams-of-dna-using-python/)
  - [How to compute the exact weight of single-strand DNA using Python](https://techoverflow.net/2025/06/22/how-to-compute-the-exact-weight-of-single-strand-dna-using-python/)
  - [How long can quarter-wavelength stubs on PCBs actually be?](https://techoverflow.net/2025/11/18/how-long-can-quarter-wavelength-stubs-on-pcbs-actually-be/)
  - [The correlation of impedance and reactance for common-mode chokes](https://techoverflow.net/2025/12/09/the-correlation-of-impedance-and-reactance-for-common-mode-chokes/)
  - [Plotting microstrip impedance vs width using UliEngineering](https://techoverflow.net/2025/12/11/plotting-microstrip-impedance-vs-width-using-uliengineering/)
  - [How to unwrap encoder values or angles which wrap at a certain numeric point](https://techoverflow.net/2025/12/18/how-to-unwrap-encoder-values-or-angles-which-wrap-at-a-certain-numeric-point/)
  - [How to generate periodic ramps in Python using UliEngineering](https://techoverflow.net/2025/12/19/how-to-generate-periodic-ramps-in-python-using-uliengineering/)
  - [Even simpler buck regulator calculation using Python](https://techoverflow.net/2026/05/01/even-simpler-buck-regulator-calculation-using-python/)
  - [Comparing simple and Shockley diode models for capacitor charging in Python using UliEngineering](https://techoverflow.net/2026/05/03/comparing-simple-and-shockley-diode-models-for-capacitor-charging-in-python-using-uliengineering/)
  - [How to compute R/C time constant in Python using UliEngineering](https://techoverflow.net/2026/05/03/how-to-compute-r-c-time-constant-in-python-using-uliengineering/)
  - [How to compute capacitor charge/discharge time through a resistor using Python](https://techoverflow.net/2026/05/03/how-to-compute-capacitor-charge-discharge-time-through-a-resistor-using-python/)
  - [How to compute diode current using the Shockley equation in Python using UliEngineering](https://techoverflow.net/2026/05/03/how-to-compute-diode-current-using-the-shockley-equation-in-python-using-uliengineering/)
  - [How to compute diode power dissipation using the Shockley equation in Python using UliEngineering](https://techoverflow.net/2026/05/03/how-to-compute-diode-power-dissipation-using-the-shockley-equation-in-python-using-uliengineering/)
  - [How to compute diode saturation current using the Shockley equation in Python using UliEngineering](https://techoverflow.net/2026/05/03/how-to-compute-diode-saturation-current-using-the-shockley-equation-in-python-using-uliengineering/)
  - [How to compute diode small-signal resistance in Python using UliEngineering](https://techoverflow.net/2026/05/03/how-to-compute-diode-small-signal-resistance-in-python-using-uliengineering/)
  - [How to compute diode thermal voltage in Python using UliEngineering](https://techoverflow.net/2026/05/03/how-to-compute-diode-thermal-voltage-in-python-using-uliengineering/)
  - [How to compute diode voltage using the Shockley equation in Python using UliEngineering](https://techoverflow.net/2026/05/03/how-to-compute-diode-voltage-using-the-shockley-equation-in-python-using-uliengineering/)
  - [How to plot Shockley diode current in Python using UliEngineering](https://techoverflow.net/2026/05/03/how-to-plot-shockley-diode-current-in-python-using-uliengineering/)

## Testing

In order to run the unit tests, first install tox:

```sh
pip install --user tox
```

and then just run it in the root directory of the cloned repository

```sh
tox
```
