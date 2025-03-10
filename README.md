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

## Classification Subgroup:
- Ketki Chakradeo
- Marco Sanchez
- Jun Hwang
- Ryan Doh

## Description
The goal of this project is to develop parameters that will be extracted from the time series data provided to us publicly by the Majorana Demonstrator experiement in order to propose machine learning models that meet the requirements outlined in the Neutrino Physics and Machine Learning (NPML) instructions included in the Majorana Demonstrator data release notes. Models in this repository include CatBoost, XG Boost, LightGBM and Random Forest.

## Installation Instructions
How to clone the repository:
``` bash
git clone https://github.com/matthewsegovia/MajoranaNeutrinoHunt.git
``` 
In order to clone the dependencies needed for our project, follow the next steps. Make sure you have Anaconda installed.<br><br>
### Anaconda Environemnt Instructions
#### 1. Replace `name_of_environment` with a name you like:
``` bash
conda env create -f environment.yml --name name_of_environment
```
#### 2. Activate the environment:
``` bash
conda activate name_of_environment
```
### Download the Proprocessed Dataset or Preprocess your own raw Data:
#### Download the preprocessed dataset:
Download the preprocessed data from this [link](https://drive.google.com/drive/folders/1SnmQemcXWPvKvJBmGkd0hSqTQ8gbs0C4), place all the csv files in the same directory as the cloned repository before running the master.py file found under the src folder.

## Features
This repository contains the files for all parameters functions that were used to build and train machine learning models

The parameters used in this investigation include:

- **Drift Time** (tdrift.py): The time taken from the initiation of charge generation to the collection at the detector's point contact at increments of 10%, 50% and 99.9%.

- **Late Charge** (lq80.py): The amount of energy being collected after 80% of the peak. 

- **Late Charge Slope** (Area Growth Rate (agr.py)): The integrated drift time of the charge collected after 80% of the waveform. 

- **Second derivative Inflection Points** (inflection.py): The amount of inflection points from 80% of our charge to the peak. 

- **Rising Edge Slope** (rising_edge.py): The slope of the charge that was recorded.

- **Rising Edge Asymmetry** (rea.py): This function measures how tilted in a direction the rising edge of the signal is.

- **Current Amplitude** (current_amplitude.py): The peak rate of charge collection, defined as I = dq/dt which means current amplitude is the derivative of charge.

- **Energy Peak** (peakandtailslope.py): The maximum analog-to-digital (ADC) count. The height of this peak correlates with the energy deposited by the particle in the detector.

- **Tail Slope** (peakandtailslope.py): The rate of charge collection over the length of the waveform’s tail. It indicates how quickly charge dissipates in the detector after the initial interaction.

- **Delayed Charge Recovery** (dcr.py): The rate of area growth in the tail slope region. This is measured by the area above the tail slope to the peak of the rise. 

- **Fourier Transform and Low Frequency Power Ratio** (fourier_lfpr.py): The Fourier Transform is a mathematical operation that transforms a time-domain signal into its frequency-domain representation. Low Frequency Power Ratio (LFPR) is used, quantifying how much of the signal’s energy is concentrated in the low-frequency threshold by the total power spectrum of the Fourier transformed waveform.  

The Master.py file combines all these parameters into one file. removedupes.py removes all duplicate rows in the processed files. 

## Data Analysis
Each folder in the repositiory contains a data model that is either a classification or a regression model. The folders contain the code as well as the results of each model.

## File Explanation
root/
- src/
  - parameter-functions/
    - Parameter extraction files needed for Mater.py
    - Data/
      - npml_cut.csv: Classification result from B10-2 (We are B10-1)
- README.md
- Analysis_Unidoc.pdf: Copy of our report
- environment.yml: Anaconda Environment file
- requirements.txt: requirements file


## Further Reading
[Majorana Demonstrator Data Release Notes](https://arxiv.org/pdf/2308.10856)
