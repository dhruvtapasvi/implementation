import pickle

history = pickle.load(open("./modelTrainingHistory/norb_pca_500_512_4_128_0_fitted_variance.history.p", "rb"))

print(history)
