## An Introduction to Discrete Variational Autoencoders

This is a simple implementation of a *discrete variational autoencoder* corresponding to our tutorial on [arxiv](https://arxiv.org/abs/2505.10344). The code follows the summarized recipe provided in Section 8 of the text. Specifically, it trains a simple MLP architecture with ReLU activations for both the encoder and decoder on a binarized version of the MNIST dataset. The latent space consists of 4 categorical latent variables with 8 categories each. Stochastic gradient descent and the Adam optimizer is used to maximize the *evidence lower bound* (ELBO) of the training data. The only package requirements are `pytorch` and `matplotlib`.

To run the main training script, run the following command.
```
python main.py
```

This should produce a plot similar to this showing the ELBO increasing throughout training. 

![alt text](https://github.com/alanjeffares/discreteVAE/blob/main/figures/elbo.png?raw=true)

Once trained, we can further explore the model by visualizing some of its generated samples. The following command runs some basic visualizations.
```
python visualize.py
```

This produces two figures. Firstly, it passes five test examples through the autoencoder showing probabilities and samples of both the input images and the reconstruction from the VAE. 

![alt text](https://github.com/alanjeffares/discreteVAE/blob/main/figures/reconstructions.png?raw=true)

Secondly, it visualizes a subset of the latent space. This is achieved by creating a single latent sample and showing the output produced as we augment each of the latent variables individually and pass the resulting latent sample through the decoder. The first row, labeled "Cat 1", is our baseline latent and we plot the mode of the distribution of the corresponding image to each latent.

![alt text](https://github.com/alanjeffares/discreteVAE/blob/main/figures/latents.png?raw=true)
