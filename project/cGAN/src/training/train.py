gan = Pix2Pix(DATASET)
gan.train(epochs=200, batch_size=1, sample_interval=200)
