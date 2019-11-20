import scipy.stats as stats
from collections import OrderedDict
import inspect
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


from pyactup import *

#The aim here is to create a random data set
#then create some random chunks to test the blend
#and test the variance on the salience (which we should know from ground truth)
#what we want to test is how sensitive it is to diversion of the probes to the mean
#so, we can chose a MU at the (known) mean (because the mu of the data set is known)
#and change the SD of the

def create_truncnorm_distribution(low, high, mu, sigma, size):
    distribution = stats.truncnorm((low - mu) / sigma, (high - mu) / sigma, loc=mu, scale=sigma)
    values = distribution.rvs(size)
    return values

def curate_function_chunks(n=0, func=0, distribution='uniform', distribution_args={},chunks=True, vectors=0):
    #Create a truncated normal distribution around mu and sigma
    chunk_data = []
    distribution_map = {'uniform':np.random.uniform,
                        'truncnorm':create_truncnorm_distribution}

    #values = distribution_map[distribution](**distribution_args)
    #feature_distribution = stats.truncnorm((min - mu) / sigma, (max - mu) / sigma, loc=mu, scale=sigma)
    # plot_values = feature_distribution.rvs(1000)
    # fig, ax = plt.subplots(1,1)
    # x = np.linspace(stats.truncnorm.ppf(0.01, min, max), stats.truncnorm.ppf(0.99, min, max), 100)
    # ax.plot(x, feature_distribution.pdf(x), 'k-', lw=2, label='frozen')
    # vals = stats.truncnorm.ppf([0.001, 0.5, 0.999], min, max)
    # np.allclose([0.001, 0.5, 0.999], stats.truncnorm.cdf(vals, min, max))
    # r = feature_distribution.rvs(5000)
    # ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    # plt.show()
    # params = list(inspect.signature(func).parameters)
    # factor_values = {}
    # for factor in params:
    #     factor_values[factor] = feature_distribution.rvs(n).tolist()
    # print("here")
    #HOW MANY factors are there?
    params = list(inspect.signature(func).parameters)
    n_factors = len(params)
    distribution_args['size'] = (n,n_factors)
    factor_values = distribution_map[distribution](**distribution_args)
    if chunks:
        for i in range(n):
            chunk = dict(zip(params,factor_values[i]))
            chunk['t'] = func(*factor_values[i].tolist())
            chunk_data.append(chunk)
    return chunk_data

def curate_probes(n=1000, func=lambda f1, f2, f3: f1 + f2 + f3,distribution='truncnorm', distribution_args={'low':-1, 'high': 1, 'mu':0,'sigma':0.1,}, chunks=True):
    chunk_data = []
    params = list(inspect.signature(func).parameters)
    n_factors = len(params)
    distribution_map = {'uniform': np.random.uniform,
                        'truncnorm': create_truncnorm_distribution}
    distribution_args['size'] = (n,n_factors)
    factor_values = distribution_map[distribution](**distribution_args)
    if chunks:
        for i in range(n):
            chunk = dict(zip(params, factor_values[i]))
            chunk_data.append(chunk)
    return chunk_data




if __name__ == "__main__":
    funct = lambda f0, f1, f2: f0 + f1 + f2 #the function I want the data to represent
    data = curate_function_chunks(n=1000, func=funct,distribution='uniform', distribution_args={'low':-1, 'high':1},chunks=True)
    probes = curate_probes(n=1000, func=funct, distribution='truncnorm', distribution_args={'low':-1, 'high':1, 'mu':0, 'sigma':0.1}, chunks=True)
    print("here")
