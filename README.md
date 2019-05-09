# UliEngineering

[![Build Status](https://travis-ci.org/ulikoehler/UliEngineering.svg?branch=master)](https://travis-ci.org/ulikoehler/UliEngineering) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/d30c78429f554aedafa147c13b443982)](https://www.codacy.com/app/ulikoehler/UliEngineering?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ulikoehler/UliEngineering&amp;utm_campaign=Badge_Grade) [![Code Climate coverage](https://codeclimate.com/github/ulikoehler/UliEngineering/badges/coverage.svg)](https://codeclimate.com/github/ulikoehler/UliEngineering/coverage) [![Code Climate](https://codeclimate.com/github/ulikoehler/UliEngineering/badges/gpa.svg)](https://codeclimate.com/github/ulikoehler/UliEngineering) ![License](https://img.shields.io/github/license/ulikoehler/UliEngineering.svg) 

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

```
sudo pip3 install UliEngineering
```

In order to install the latest bleeding-edge version, use:

```
sudo pip3 install git+https://github.com/ulikoehler/UliEngineering.git
```

We *highly recommend* you also install scipy:
```
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
