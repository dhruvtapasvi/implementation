import matplotlib
matplotlib.use('Agg')


from experiments.SimpleMnistExperiment import SimpleMnistExperiment


if __name__ == '__main__':
    simpleMnistExperiment = SimpleMnistExperiment()
    simpleMnistExperiment.run()
