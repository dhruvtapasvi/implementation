# Interpolating in the Latent Space of Variational Autoencoders
Given a vector x and y from a larger dataset, is it possible to create a vector z with properties intermediate to those of x and y? For example, if x is an image of an object taken from an angle of a degrees and y is an image of the same object taken from an angle of b degrees, can we create an image z of the object taken from an angle between a and b? I seek to do this by using training a variational autoencoder, and performing interpolation in its latent space.

## Interpreter Settings
### Python Version
3.6.3
### Dependencies
See requirements.txt

## Setup and Testing
### Setup
`pip install -r requirements.txt`
### Running the Code
*src* contains the main bits of code for the project  
At the top level of this folder, there are a number of Python scripts which can be executed directly e.g. to produce a dataset or to generate the results for the dissertation  
To run such a script, simply run `python3 -m <script>` from the *src* folder
### Test
`nosetests`

## Project Structure
*src* contains the main package for the code.

*test* contains tests for the modules present in *src*.

*rough* contains some scripts for quickly testing the code in *src*. These may be broken at any point so please do not rely on them.

*scripts* contains some useful scripts e.g. for downloading datasets which are too large to store in the repository. Run scripts from the implementation folder.

*res* contains resources useful for the project, e.g. datasets. If a resource is too large, please consider adding a script/program to reproduce it instead of checking it into the repository.
