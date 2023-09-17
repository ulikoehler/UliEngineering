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
  - [Normalizing electronics engineering value notations using Python](https://techoverflow.net/2015/06/09/normalizing-electronics-engineering-value-notations-using-python/)
  - [Finding the nearest E96 resistor value in Python](https://techoverflow.net/2015/05/19/finding-the-nearest-e96-resistor-value-in-python/)
  - [Easy zero crossing detection in Python using UliEngineering](https://techoverflow.net/2018/12/31/easy-zero-crossing-detection-in-python-using-uliengineering/)
  - [Easily generate sine/cosine wave test data in Python using UliEngineering](https://techoverflow.net/2018/12/31/easily-generate-sine-cosine-wave-data-in-python-using-uliengineering/)
  - [Easily generate square/triangle/sawtooth/inverse sawtooth waveform data in Python using UliEngineering](https://techoverflow.net/2018/12/31/easily-generate-square-triangle-sawtooth-inverse-sawtooth-waveform-data-in-python-using-uliengineering/)
  - [Easily compute & visualize FFTs in Python using UliEngineering](https://techoverflow.net/2018/12/31/easily-compute-visualize-ffts-in-python-using-uliengineering/)
  - [Computing resistor power dissipation in Python using UliEngineering](https://techoverflow.net/2019/05/09/computing-the-temperature-under-load-of-your-electronics-components-in-python-using-uliengineering/)
  - [Computing crystal load capacitance using Python & UliEngineering](https://techoverflow.net/2019/05/10/computing-crystal-load-capacitance-using-python-uliengineering/)
  - [How to convert Celsius/Fahrenheit/Kelvin temperatures in Python using UliEngineering](https://techoverflow.net/2019/05/10/how-to-convert-celsius-fahrenheit-kelvin-temperatures-in-python-using-uliengineering/)
  - [How to get unit/resolution of NumPy np.timedelta64 object](https://techoverflow.net/2019/05/13/how-to-get-unit-resolution-of-numpy-np-timedelta64-object/)
  - [How to get unit/resolution of NumPy np.datetime64 object](https://techoverflow.net/2019/05/13/how-to-get-unit-resolution-of-numpy-np-datetime64-object/)
  - [How to iterate all days of year using Python](https://techoverflow.net/2019/05/16/how-to-iterate-all-days-of-year-using-python/)
  - [How to get number of days in month in Python](https://techoverflow.net/2019/05/16/how-to-get-number-of-days-in-month-in-python/)
  - [How to compute candelas from lumens by apex angle](https://techoverflow.net/2019/08/19/lumen-to-candela-online-calculator-python-code/)
  - [How to compute resistor voltage divider ratio using Python](https://techoverflow.net/2021/12/16/how-to-compute-resistor-voltage-divider-ratio-using-python/)
  - [How to generate datetime for every hour on a given day in Python](https://techoverflow.net/2022/06/19/how-to-generate-datetime-for-every-hour-on-a-given-day-in-python/)
  - [How to generate datetime for every minute on a given day in Python](https://techoverflow.net/2022/06/19/how-to-generate-datetime-for-every-minute-on-a-given-day-in-python/)
  - [How to generate datetime for every second on a given day in Python](https://techoverflow.net/2022/06/19/how-to-generate-datetime-for-every-second-on-a-given-day-in-python/)
  - [How to compute MRI Larmor frequency for a given magnetic field using Pythonf](https://techoverflow.net/2023/02/04/how-to-compute-mri-larmor-frequency-for-a-given-magnetic-field-using-python/)
  - [How to compute voltage divider output voltage using Python](https://techoverflow.net/2023/02/05/how-to-compute-voltage-divider-output-voltage-using-python/)
  - [How to format axis as dB (decibel) using matplotlib](https://techoverflow.net/2023/03/13/how-to-format-axis-as-db-decibel-using-matplotlib/)
  - [How to compute Buck/Boost/LDO output voltage by feedback resistors using Python](https://techoverflow.net/2023/04/09/how-to-compute-buck-boost-ldo-output-voltage-by-feedback-resistors-using-python/)
  - [How to compute the weight of a titanium or stainless steel rod using UliEngineering in Python](https://techoverflow.net/2023/05/21/how-to-compute-the-weight-of-a-titanium-or-stainless-steel-rod-using-uliengineering-in-python/)
  - [How to compute non-inverting OpAmp amplifier gain using UliEngineering in Python](https://techoverflow.net/2023/09/18/how-to-compute-non-inverting-opamp-amplifier-gain-using-uliengineering-in-python/)

## Testing

In order to run the unit tests, first install tox:

```sh
pip install --user tox
```

and then just run it in the root directory of the cloned repository

```sh
tox
```
