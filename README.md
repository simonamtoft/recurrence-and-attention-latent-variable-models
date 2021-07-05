# Recurrence and Attention in Latent Variable Models
In recent years deep latent variable models have been widely used for image generation and representation learning. Standard approaches employ shallow inference models with restrictive mean-field assumptions.  A way to increase inference expressivity is to define a hierarchy of latent variables in space and build structured approximations. Using this approach the size of the model grows linearly in the number of layers.

An orthogonal approach is to define hierarchies in time using a recurrent model. This approach exploits parameter sharing and gives us the possibility to define models with infinite depth (assuming a memory-efficient learning algorithm).

In this project, we study recurrent latent variable models for image generation. We focus on attentive models, i.e. models that use attention to decide where to focus on and what to update, refining their output with a sequential update. This is done by implementing the DRAW model, which is described in [the DRAW paper](https://arxiv.org/abs/1502.04623), both with basic and filterbank attention. The performance of the implemented DRAW model is then compared to both a standard VAE and a LadderVAE implementation.

The project is carried out by [Simon Amtoft Pedersen](https://github.com/simonamtoft), and supervised by Giorgio Giannone.

## Variational Autoencoder
Variational Autoencoders (VAEs) are a type of latent variable model that can be used for generative modelling. The VAEs consists of a decoder part and an encoder part, that is trained by optimizing the Evidence Lower Bound (ELBO). The generative model is given by <img src="https://latex.codecogs.com/svg.image?\inline&space;p_\theta(z)&space;=&space;p_\theta(x|z)&space;p_\theta(z)" title="\inline p_\theta(z) = p_\theta(x|z) p_\theta(z)" /> and the samples are then drawn from the distribution by <img src="https://latex.codecogs.com/svg.image?\inline&space;z&space;\sim&space;p_\theta(z|x)&space;=&space;\frac{p_\theta(x|z)&space;p_\theta(z)}{p_\theta(x)}&space;" title="\inline z \sim p_\theta(z|x) = \frac{p_\theta(x|z) p_\theta(z)}{p_\theta(x)} " />. The objective is then to optimize <img src="https://latex.codecogs.com/svg.image?\inline&space;\sum_i&space;\mathcal{L_{\theta,\phi}}(x_i)" title="\inline \sum_i \mathcal{L_{\theta,\phi}}(x_i)" /> where ELBO is given as <img src="https://latex.codecogs.com/svg.image?\inline&space;\mathcal{L_{\theta,\phi}}(x)&space;=&space;\mathbb{E}_{q_\phi(z|x)}[\log&space;p_\theta&space;(x|z)]&space;&plus;&space;\mathbb{E}_{q_\phi(z|x)}\left[\log\frac{p_\theta(z)}{q_\phi(z|x)}\right]" title="\inline \mathcal{L_{\theta,\phi}}(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta (x|z)] + \mathbb{E}_{q_\phi(z|x)}\left[\log\frac{p_\theta(z)}{q_\phi(z|x)}\right]" />.

Once the model is trained, it can generate new examples by sampling <img src="https://latex.codecogs.com/svg.image?\inline&space;z&space;\sim&space;N(z|0,1)" title="\inline z \sim N(z|0,1)" /> and then passing this sample through the decoder to generate a new example <img src="https://latex.codecogs.com/svg.image?\inline&space;x&space;\sim&space;N(x|\mu(z),&space;diag(\sigma^2(z)))" title="\inline x \sim N(x|\mu(z), diag(\sigma^2(z)))" />.


### Ladder VAE
An extension of the standard VAE is the [Ladder VAE](https://arxiv.org/pdf/1602.02282.pdf), which adds sharing of information and parameters between the encoder and decoder by splitting the latent variables into L layers, such that the model can be described by:

<img src="https://latex.codecogs.com/svg.image?\inline&space;p_\theta(z)&space;=&space;p_\theta(z_L)\prod_{i=1}^{L-1}&space;p_\theta(z_i&space;|z_{i&plus;1})&space;" title="\inline p_\theta(z) = p_\theta(z_L)\prod_{i=1}^{L-1} p_\theta(z_i |z_{i+1}) " />

<img src="https://latex.codecogs.com/svg.image?\inline&space;p_\theta(z_i&space;|&space;z_{i&plus;1})&space;=&space;N(z_i|&space;\mu_{p,i},&space;\sigma^2_{p,i}),&space;\;\;\;\;&space;p_\theta(z_L)&space;=&space;N(z_L|0,I)" title="\inline p_\theta(z_i | z_{i+1}) = N(z_i| \mu_{p,i}, \sigma^2_{p,i}), \;\;\;\; p_\theta(Z_L) = N(z_L|0,I)" />

<img src="https://latex.codecogs.com/svg.image?p_\theta(x|z_1)&space;=&space;N(x|\mu_{p,0},\sigma^2_{p,0})" title="p_\theta(x|z_1) = N(x|\mu_{p,0},\sigma^2_{p,0})" />


## Deep Recurrent Attentive Writer
The Deep Recurrent Attentive Writer (DRAW) model is a VAE like model, trained with stochastic gradient descent, proposed in the [original DRAW paper](https://arxiv.org/pdf/1502.04623.pdf). The main difference is, that the DRAW model iteratively generates the final output instead of doing it in a single shot like a standard VAE. Additionally, the encoder and decoder uses recurrent networks instead of standard linear networks.

### The Network
The model goes through T iterations, where we denote each time-step iteration by t. When using a diagonal Gaussian for the latent distribution, we have:

<img src="https://latex.codecogs.com/svg.image?\mu_t&space;=&space;W(h_t^{enc}),&space;\;\;\;&space;\;\;&space;\sigma_t&space;=&space;\exp(W(h_t^{enc}))" title="\mu_t = W(h_t^{enc}), \;\;\; \;\; \sigma_t = \exp(W(h_t^{enc}))" />

Samples are then drawn from the latent distribution <img src="https://latex.codecogs.com/svg.image?z_t&space;\sim&space;Q(z_t|h_t^{enc})" title="z_t \sim Q(z_t|h_t^{enc})" />, which we pass to the decoder, which outputs <img src="https://latex.codecogs.com/svg.image?h_t^{dec}" title="h_t^{dec}" /> that is added to the canvas, <img src="https://latex.codecogs.com/svg.image?c_t" title="c_t" />, using the write operation. At each time-step, <img src="https://latex.codecogs.com/svg.image?t&space;=&space;1,...,T" title="t = 1,...,T" />, we compute:

<img src="https://latex.codecogs.com/svg.image?\!\!\!\!\!\!\!\!\!&space;\hat{x}_t&space;=&space;x&space;-&space;\sigma(c_{t-1})\\r_t&space;=&space;read(x_t,\hat{x}_t,h_{t-1}^{dec})\\h_t^{enc}&space;=&space;RNN^{enc}(h_{t-1}^{enc},&space;[r_t,&space;h_{t-1}^{dec}]])\\z_t&space;\sim&space;Q(z_t|h_t^{enc})\\h_t^{dec}&space;=&space;RNN^{dec}(h_{t-1}^{dec},&space;z_t)\\c_t&space;=&space;c_{t-1}&space;&plus;&space;write(h_t^{dec})&space;" title="\!\!\!\!\!\!\!\!\! \hat{x}_t = x - \sigma(c_{t-1})\\r_t = read(x_t,\hat{x}_t,h_{t-1}^{dec})\\h_t^{enc} = RNN^{enc}(h_{t-1}^{enc}, [r_t, h_{t-1}^{dec}]])\\z_t \sim Q(z_t|h_t^{enc})\\h_t^{dec} = RNN^{dec}(h_{t-1}^{dec}, z_t)\\c_t = c_{t-1} + write(h_t^{dec}) " />

### Data Generation
Generating images from the model is then done by iteratively picking latent samples from the prior distribution, and updating the canvas with the decoder:

<img src="https://latex.codecogs.com/svg.image?\!\!\!\!\!\!\!\!\!\tilde{z}_t&space;\sim&space;p(z_t)\\\tilde{h}_t^{dec}&space;=&space;RNN^{dec}(\tilde{h}_{t-1}^{dec},\tilde{z})\\\tilde{c}_t&space;=&space;\tilde{c}_{t-1}&space;&plus;&space;write(\tilde{h}_t^{dec})\\\tilde{x}&space;\sim&space;D(X|\tilde{c}_T)&space;" title="\!\!\!\!\!\!\!\!\!\tilde{z}_t \sim p(z_t)\\\tilde{h}_t^{dec} = RNN^{dec}(\tilde{h}_{t-1}^{dec},\tilde{z})\\\tilde{c}_t = \tilde{c}_{t-1} + write(\tilde{h}_t^{dec})\\\tilde{x} \sim D(X|\tilde{c}_T) " />

### Read and Write operations
Finally we have the read and write operations. These can be used both with and without attention.

In the version without attention, the entire input image is passed to the encoder for every time-step, and the decoder modifies the entire canvas at every step. The two operations are then given by

<img src="https://latex.codecogs.com/svg.image?\!\!\!\!\!\!\!\!read(x,&space;\hat{x}_t,&space;h_{t-1}^{dec})&space;=&space;[x,&space;\hat{x}_t]\\write(h_t^{dec})&space;=&space;W(h_t^{dec})&space;" title="\!\!\!\!\!\!\!\!read(x, \hat{x}_t, h_{t-1}^{dec}) = [x, \hat{x}_t]\\write(h_t^{dec}) = W(h_t^{dec}) " />

In oder to use attention when reading and writing, a two-dimensional attention form is used with an array of two-dimensional Gaussian filters. For an input of size A x B, the model generates five parameters from the output of the decoder, which is used to compute the grid center, stride and mean location of the filters:

<img src="https://latex.codecogs.com/svg.image?\!\!\!\!\!\!\!\!\!(\tilde{g}_X,&space;\tilde{g}_Y,&space;\log&space;\sigma^2,&space;\log&space;\tilde{\delta},&space;\log&space;\gamma)&space;=&space;W(h^{dec}_t)\\g_X&space;=&space;\frac{A&plus;1}{2}(\tilde{g}_X&space;&plus;&space;1)\\g_X&space;=&space;\frac{A&plus;1}{2}(\tilde{g}_X&space;&plus;&space;1)\\&space;\delta&space;=&space;\frac{\max(A,B)&space;-&space;1}{N&space;-&space;1}&space;\tilde{\delta}\\\mu_X^i&space;=&space;g_X&space;&plus;&space;(i&space;-&space;N/2&space;-&space;0.5)&space;\delta\\\mu_Y^j&space;=&space;g_Y&space;&plus;&space;(j&space;-&space;N/2&space;-&space;0.5)&space;\delta&space;" title="\!\!\!\!\!\!\!\!\!(\tilde{g}_X, \tilde{g}_Y, \log \sigma^2, \log \tilde{\delta}, \log \gamma) = W(h^{dec}_t)\\g_X = \frac{A+1}{2}(\tilde{g}_X + 1)\\g_X = \frac{A+1}{2}(\tilde{g}_X + 1)\\ \delta = \frac{\max(A,B) - 1}{N - 1} \tilde{\delta}\\\mu_X^i = g_X + (i - N/2 - 0.5) \delta\\\mu_Y^j = g_Y + (j - N/2 - 0.5) \delta " />

## Repo Structure
In this repo you will find the three different model classes in the models directory, and the necessary training loops for each model is found in the training directory.
Additionally the attention, encoder and decoder, and other modules used in these models can be found in the layers directory.

