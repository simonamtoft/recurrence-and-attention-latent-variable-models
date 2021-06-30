# Recurrence and Attention in Latent Variable Models
In recent years deep latent variable models have been widely used for image generation and representation learning. Standard approaches employ shallow inference models with restrictive mean-field assumptions.  A way to increase inference expressivity is to define a hierarchy of latent variables in space and build structured approximations. Using this approach the size of the model grows linearly in the number of layers.

An orthogonal approach is to define hierarchies in time using a recurrent model. This approach exploits parameter sharing and gives us the possibility to define models with infinite depth (assuming a memory-efficient learning algorithm).

In this project, we study recurrent latent variable models for image generation. We focus on attentive models, i.e. models that use attention to decide where to focus on and what to update, refining their output with a sequential update. This is done by implementing the DRAW model, which is described in [the DRAW paper](https://arxiv.org/abs/1502.04623), both with basic and filterbank attention. The performance of the implemented DRAW model is then compared to both a standard VAE and a LadderVAE implementation.

The project is carried out by [Simon Amtoft Pedersen](https://github.com/simonamtoft), and supervised by Giorgio Giannone.

## Structure
In this repo you will find the three different model classes in the models directory, and the necessary training loops for each model is found in the training directory.
Additionally the attention, encoder and decoder, and other modules used in these models can be found in the layers directory.

