# Majorana Neutrino Hunt
Capstone Project B10

Contributors:
- Matthew Segovia
- Jun Hwang
- Ryan Doh
- Haotian Zhu
- Ammie Xie
- Ketki Chakradeo
- Marco Sanchez

## Description
The goal of this project is to develop parameters that will be extracted from the time series data provided to us publicly by the Majorana Demonstrator experiement in order to propose machine learning models that meet the requirements outlined in the Neutrino Physics and Machine Learning (NPML) instructions included in the Majorana Demonstrator data release notes.

## Installation Instructions
How to clone the repository:
``` bash
git clone https://github.com/matthewsegovia/MajoranaNeutrinoHunt.git
``` 

## Features
This repository contains the files for all parameters functions that will be used to build and train machine learning models 

The parameters used in this investigation include:

- Drift Time (tdrift: 10%, 50%, 99.9%): The time taken from the initiation of charge generation to the collection at the detector's point contact at increments of 10%, 50% and 99.9%.
- Late Charge (LQ80): The amount of energy being collected after 80% of the peak.
- Late Charge Slope: The integrated drift time of the charge collected after 80% of the waveform.
- Second derivative Inflection Points: The amount of inflection points from 80% of our charge to the peak.
- Rising Edge Slope: 
- Rising Edge Asymmetry (REA): 
- Current Amplitude: 
- Tail Slope: 
- Delayed Charge Recovery (DCR): The area above the tail slope to the peak of the rise.
- Fourier Transform (LFPR): 
- Energy Peak: 
- Area Growth Rate (AGR): 

The Master.py file contains all these parameters combined and requriements.txt file lists all the python packages needed prior to running the code.

## Contributing

## Further Reading
