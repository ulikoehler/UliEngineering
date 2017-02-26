# UliEngineering

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d30c78429f554aedafa147c13b443982)](https://www.codacy.com/app/ulikoehler/UliEngineering?utm_source=github.com&utm_medium=referral&utm_content=ulikoehler/UliEngineering&utm_campaign=badger)
[![Build Status](https://travis-ci.org/ulikoehler/UliEngineering.svg?branch=master)](https://travis-ci.org/ulikoehler/UliEngineering) [![Code Climate coverage](https://codeclimate.com/github/ulikoehler/UliEngineering/badges/coverage.svg)](https://codeclimate.com/github/ulikoehler/UliEngineering/coverage) [![Code Climate](https://codeclimate.com/github/ulikoehler/UliEngineering/badges/gpa.svg)](https://codeclimate.com/github/ulikoehler/UliEngineering)

A python library for calculations perfomed in electronics engineering.

This project is still in pre-alpha and only contains a collection of electronics functionality from [techoverflow.net](http://techoverflow.net). The python package itself has not been built or published yet.

Pull requests and bugreports of any kind are happily accepted.

## Installation

Run this command on your favourite shell:

```
sudo pip3 install git+https://github.com/ulikoehler/UliEngineering.git
```

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

[Normalizing electronics engineering value notations using Python](https://techoverflow.net/2015/06/09/normalizing-electronics-engineering-value-notations-using-python/)
[Finding the nearest E96 resistor value in Python](https://techoverflow.net/2015/05/19/finding-the-nearest-e96-resistor-value-in-python/)
