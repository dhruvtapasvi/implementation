from experiment.Experiment import Experiment


class InterpolateExperiment(Experiment):
    def run(self):
        """
        TODO:
        Given left dataset, right dataset and autoencoder perform interpolation in latent space and in image space
        Print reconstructions as (Left, Interp, Right)
        Given controls CORRECT and COMPLETELY_INCORRECT (e.g. extrapolated beyond range), also construct above
        Then compare MET(Interp, CORRECT) and MET(Interp, COMPLETELY_INCORRECT)
        Variation: MET in { MSE-IS, MSE-LS, MBCE-IS)
        Variation: AUTOENC in { DENSE, CONV, PCA-DENSE, CVAE, IDENTITY }
        Variation: DATASET
        This experiment can loop over all metrics (first factor of variation
        The outer experiment can instantiate this one many times with the correct autoencoder
        (maybe a method to run with the autoencoder)
        What about reporting per class? Need to think about that in more details
        """
        pass
