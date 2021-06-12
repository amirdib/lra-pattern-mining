import numpy as np
#import tensorflow as tf
#import tensorflow_probability as tfp

#tfd = tfp.distributions


# def make_Multivariate_Bernoulli(theta):
#     return tfd.Independent(tfd.Bernoulli(probs=theta),
#                            reinterpreted_batch_ndims=1)


# def make_random_noise(p0, d, N):
#     theta = d * [p0]
#     return make_Multivariate_Bernoulli(theta).sample(N)


# def bernoulliMixture_make_likelihood(theta, mix_probs):
#     y = tfd.MixtureSameFamily(
#         mixture_distribution=tfd.Categorical(
#             probs=mix_probs),
#         components_distribution=tfd.Independent(
#             distribution=tfd.Bernoulli(probs=theta),
#             reinterpreted_batch_ndims=1))
#     return y


# def make_mixture_bern(theta, mix_probs, d, N):
#     observations = bernoulliMixture_make_likelihood(theta, mix_probs).sample(N)
#     observations = tf.cast(observations, tf.float32)
#     return observations

# def make_Multivariate_Bernoulli(theta):
#     return tfd.Independent(tfd.Bernoulli(probs=theta),
#                            reinterpreted_batch_ndims=1)

def make_rademacher_variable(m):
    return (2*np.random.binomial(n=1, p=.5, size=m) - 1)
