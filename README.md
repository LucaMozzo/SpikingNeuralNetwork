# Stochastic Spiking Neural Network
Luca Mozzo - King's College London

![poster](http://lucamozzo.altervista.org/wp-content/uploads/2018/08/Poster.png "The poster for my thesis")


## What is this?
This repository contains the code used for the thesis of my Bachelor degree at King's College London in 2018. The code is protected under the GNU General Public License v3.0, and you're more than welcome to contribute to the project.
This code is an efficient C++ implementation of a Spiking Neural Network, which was heavily inspired by this work: [https://arxiv.org/pdf/1710.10704.pdf](https://arxiv.org/pdf/1710.10704.pdf).

## Branches and variations
The code contains 6 different branches:
- master: the revised code that I used for my dissertation, with some additions
- RateDecoding: the implementation using rate decoding, similar to the one in master
- RateDecodingQuantization: same as above, but it has extra code for hardware simulations (quantization)
- RateDecodingQuantNewMethod: some rate decoding-based experiments - not working
- FTS: first-to-spike method implementation
- FTSQuantization: first-to-spike method implementation with extra code for hardware simulations (quantization)

## Bugs and known problems
In recent releases the RAISED_COSINE basis function is known not to work, and unfortunately I don't have time to debug it. Should you find the problem you're free to suggest a solution and I will implement it

## Setup
You need Visual Studio 2017, the MNIST database and Open CV3 installed on your machine to run the code. Use the configuration "release" to train, otherwise it takes 30m per epoch
Some code has also a Linux version, but it's not necessarily consistent (the code is 99% linux compatible anyway, there are only minor changes)

## Running it
`Constants.h` contains parameters.
`SNN.cpp` should contain your logic.
You can `ImportFile` or `ExportFile` where the function is implemented (it's quite recent), or use the SQLite export using `ExportData` and `ImportData`.
To train you would so something like
```
auto n = Network();
n.Train<0>();
```
or specify more parameters in the train function. The `<0>` is the size of the filter (you could create a filter and train/validate only on the specified digits)

To train and validate at every epoch (i.e. generate a plot)
```
auto n = Network();
n.TrainVal(200, 999999, 10000, true);
```
which trains for 200 epochs, using at most 999999 images per label (all of them in this case) and 10k images for validation, generating a report at the end.
The file `results.csv` must be created in advance, otherwise no results will be recorded

To validate
```
auto n = Network();
n.Validate<0>();
```
The other parameters are straightforward.

### This code is a mess!
The original implementation of this project took 2h37m to do a single epoch, and since it was being used for research purposes, the time was not enough to play around with all hyperparameters, so I had to make it super-efficient. All arrays are fixed-length and hence stored in the stack (if you increase the value of T you might have to increase the max stack size of a few MB!) and all matrix operations have been implemented from scratch and performance-tested.
Unfortunately, the code is not as readable as it used to be, but there is some documentation in the header files that should help you, or otherwise ask a question as a Github issue.