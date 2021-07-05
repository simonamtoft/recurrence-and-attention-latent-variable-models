# Recurrence and Attention in Latent Variable Models
In recent years deep latent variable models have been widely used for image generation and representation learning. Standard approaches employ shallow inference models with restrictive mean-field assumptions.  A way to increase inference expressivity is to define a hierarchy of latent variables in space and build structured approximations. Using this approach the size of the model grows linearly in the number of layers.

An orthogonal approach is to define hierarchies in time using a recurrent model. This approach exploits parameter sharing and gives us the possibility to define models with infinite depth (assuming a memory-efficient learning algorithm).

In this project, we study recurrent latent variable models for image generation. We focus on attentive models, i.e. models that use attention to decide where to focus on and what to update, refining their output with a sequential update. This is done by implementing the DRAW model, which is described in [the DRAW paper](https://arxiv.org/abs/1502.04623), both with basic and filterbank attention. The performance of the implemented DRAW model is then compared to both a standard VAE and a LadderVAE implementation.

The project is carried out by [Simon Amtoft Pedersen](https://github.com/simonamtoft), and supervised by Giorgio Giannone.

## Variational Autoencoder
Variational Autoencoders (VAEs) are a type of latent variable model that can be used for generative modelling. The VAEs consists of a decoder part and an encoder part, that is trained by optimizing the Evidence Lower Bound (ELBO). The generative model is given by <img src="https://latex.codecogs.com/svg.image?\inline&space;p_\theta(z)&space;=&space;p_\theta(x|z)&space;p_\theta(z)" title="\inline p_\theta(z) = p_\theta(x|z) p_\theta(z)" /> and the samples are then drawn from the distribution by <img src="https://latex.codecogs.com/svg.image?\inline&space;z&space;\sim&space;p_\theta(z|x)&space;=&space;\frac{p_\theta(x|z)&space;p_\theta(z)}{p_\theta(x)}&space;" title="\inline z \sim p_\theta(z|x) = \frac{p_\theta(x|z) p_\theta(z)}{p_\theta(x)} " />. 

Once the model is trained, it can generate new examples by sampling <img src="https://latex.codecogs.com/svg.image?\inline&space;z&space;\sim&space;N(z|0,1)" title="\inline z \sim N(z|0,1)" /> and then passing this sample through the decoder to generate a new example <img src="https://latex.codecogs.com/svg.image?\inline&space;x&space;\sim&space;N(x|\mu(z),&space;diag(\sigma^2(z)))" title="\inline x \sim N(x|\mu(z), diag(\sigma^2(z)))" />.


# Ladder VAE
An extension of the standard VAE is the [Ladder VAE](https://arxiv.org/pdf/1602.02282.pdf), which adds sharing of information and parameters between the encoder and decoder. 

## Deep Recurrent Attentive Writer



## Structure
In this repo you will find the three different model classes in the models directory, and the necessary training loops for each model is found in the training directory.
Additionally the attention, encoder and decoder, and other modules used in these models can be found in the layers directory.

