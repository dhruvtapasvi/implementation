from model.MnistDenseAutoencoder import MnistDenseAutoencoder

if __name__ == '__main__':
    print("hello")
    x = MnistDenseAutoencoder((28, 28), 256, 10)
    x.summary()
