# Bayesian-Linear-Regression-from-Scartch-in-BlackJax

We have to make Bayesian Linear Regression from scartch using BlackJax sampling method.

I started the task by searching more information abput BlackJax and Bayesian Linear Regression.

Here's what I came across:
   
   # What is BlackJax?
   It is a sampler for JAX that works both on CPU AND GPU. BlackJAX is an MCMC sampling library based on JAX. 
    BlackJAX provides well-tested and ready to use sampling algorithms: 
    It is also explicitly designed to be modular: 
    it is easy for advanced users to mix-and-match different metrics, integrators, trajectory integrations, etc.
    
    For more on BlackJax sampling : https://blackjax-devs.github.io/blackjax/examples/Introduction.html
    
   # What is Bayesian Linear Regression?
  In statistics, Bayesian linear regression is an approach to linear regression in which the statistical analysis is undertaken within the context of Bayesian inference. When the regression model has errors that have a normal distribution, and if a particular form of prior distribution is assumed, explicit results are available for the posterior probability distributions of the model's parameters.

The aim of Bayesian Linear Regression is not to find the single “best” value of the model parameters, but rather to determine the posterior distribution for the model parameters. Not only is the response generated from a probability distribution, but the model parameters are assumed to come from a distribution as well. The posterior probability of the model parameters is conditional upon the training inputs and outputs:

# Priors: 
If we have domain knowledge, or a guess for what the model parameters should be, we can include them in our model, unlike in the frequentist approach which assumes everything there is to know about the parameters comes from the data. If we don’t have any estimates ahead of time, we can use non-informative priors for the parameters such as a normal distribution.
# Posterior: 
The result of performing Bayesian Linear Regression is a distribution of possible model parameters based on the data and the prior. This allows us to quantify our uncertainty about the model: if we have fewer data points, the posterior distribution will be more spread out.
As the amount of data points increases, the likelihood washes out the prior, and in the case of infinite data, the outputs for the parameters converge to the values obtained from OLS.

Reference Link: https://en.wikipedia.org/wiki/Bayesian_linear_regression#:~:text=In%20statistics%2C%20Bayesian%20linear%20regression,the%20context%20of%20Bayesian%20inference.

https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7

The basic step of the implementation of Bayesian Linear Regression is to import the essential amd required libraries:

      pip install blackjax
      
      import jax
      



 
    
    
