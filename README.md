# Generation of audio training datasets from a simulated soundstage

The purpose of this Python module was to generate an audio training dataset from simulated acoustic scenarios to have labeled examples to train deep learning or neural network-based audio array processing algorithms for applications such as DOA estimation, source separation and others. 

## Description

The module consists of the dataSetAudio class, which is initialized with the intervals of directions of arrival, reverberation times, SNR values, distances between sources to the microphone array, and the duration of each labeled instance of the dataset.

Then, a simulation method is employed to simulate the soundstage and generate the dataset from the dataset's name and path, the audio file, the room dimensions, the source and microphone positions, and others.

It selects between two microphone array geometries: uniform linear array (ULA) and uniform circular (UCA) array and creates the dataset, a `.csv` file, meaning each value is in plain text and separated by commas.

##  Acknowledgement and Foundations

At the core of this module is the [pyroomacoustics library](#references), that allows for tuning of the rooms to polyhedral shapes, inputting the location of an arbitrary number of sources and microphones, using ray tracing to complement the ISM etc. Pyroomacoustics has a function that convolves the RIR with the audio of the sound source and adds additive white gaussian noise (AWGN) to create the simulated microphone inputs with the selected SNR. This library is found in the repositories of @LCAV.

To achieve real time application, accurate VAD is needed to feed only voiced segments to the algorithm and to prepare the dataset for simulations. [The WebRTCvad Python API](#references) was selected for this task, that was developed by Google for the Webrtc project which aids developers create real time web communication services. This library is found in the repositories of @wiseman

   ### Dependencies

   - [pyroomacoustics](https://pypi.org/project/pyroomacoustics/)
   - [webrtcvad](https://pypi.org/project/webrtcvad/)



## How to use it

1. Import the module:
   ```py
   import dataSetAudio as dsa
   ```

1. Create a `dataSetAudio` object:
   ```python
   test = dsa.dataSetAudio()
   ```

1. Give it a path and a segment of audio or an audio file. You must specify the number of microphones and the sampling frequency:
   ```py
   #number of chanels/mics
   M = 8
   #sampling frequency
   fs = 16000
   #number of directions of arrivals
   doas = 1500
   test.SimuData("dataset.csv",voice,fs,M,array_mic='UCA',random=doas)
   ```

### Examples

A couple of [notebooks](./examples/example_UCA.ipynb) are available.

A comprehensive set of examples covering most of the functionalities of the module can be found in the `examples` folder of the GitHub repository.

## References

[[1]](https://ieeexplore.ieee.org/document/8461310)
   Scheibler, R., Bezzam, E. and Dokmanić, I.: ``Pyroomacoustics: A Python Package for Audio Room Simulation and Array Processing Algorithms'', in: 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 351–355, doi: 10.1109/ICASSP.2018.8461310.

[[2]](https://pypi.org/project/webrtcvad/)
   Python interface to the Google WebRTC Voice Activity Detector (VAD), Project description.
