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

The requriements.txt file lists all the python packages as well as versions needed prior to running the code.

## Features
This repository contains the files for all parameters functions that will be used to build and train machine learning models 

The parameters used in this investigation include:

- Drift Time (tdrift.py): The time taken from the initiation of charge generation to the collection at the detector's point contact at increments of 10%, 50% and 99.9%.
<br />
- Late Charge (lq80.py): The amount of energy being collected after 80% of the peak. 
- Late Charge Slope: The integrated drift time of the charge collected after 80% of the waveform. 
- Second derivative Inflection Points (inflection.py): The amount of inflection points from 80% of our charge to the peak. 
- Rising Edge Slope (rising_edge.py): 
- Rising Edge Asymmetry (rea.py): 
- Current Amplitude (current_amplitude.py):  
- Energy Peak (peakandtailslope.py):  
- Tail Slope (peakandtailslope.py):   
- Delayed Charge Recovery (dcr.py): The area above the tail slope to the peak of the rise. 
- Fourier Transform and Low Frequency Power Ratio (fourier_lfpr.py):  
- Area Growth Rate (agr.py):  

The Master.py file combines all these parameters into one file.

## Contributing

## Further Reading
