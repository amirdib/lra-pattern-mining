import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_bound(empirical_average_range, bound_values_upper, bound_values_lower, marker,title_suffix, color='red',ax=None, alpha=.1,**kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.fill_between(empirical_average_range, bound_values_lower,bound_values_upper, color=color, alpha=.05)
    ax.plot(empirical_average_range,bound_values_upper, marker, label= title_suffix,color=color)
    ax.plot(empirical_average_range,bound_values_lower, marker) #label='Lower-' + title_suffix,color=color)
    ax.grid()    

def plot_bound_from_records(bounds_record, empirical_average_range,bounds = ['GRCMC','Bennett', 'LRCMC', 'LRCMC_corrected', 'LRCMassart', 'MCRapper', 'Union'], ax=None, **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots(1, 1)        
            
        if 'Union' in bounds:
            hoeffding_bounds = bounds_record['hoeffding_bounds']
            plot_bound(empirical_average_range=empirical_average_range,
                       bound_values_lower=empirical_average_range + hoeffding_bounds,
                       bound_values_upper=empirical_average_range - hoeffding_bounds,
                       title_suffix= 'Hoeffding',
                       marker='--+',
                       color='brown',
                       ax=ax)
            
        if 'Bennett' in bounds: 
            bennett_bound = bounds_record['bennett_bounds']
            plot_bound(empirical_average_range=empirical_average_range,
                       bound_values_lower=empirical_average_range + bennett_bound,
                       bound_values_upper=empirical_average_range - bennett_bound,
                       title_suffix= 'Bennett',
                       marker='--^',
                       color='blue',
                       ax=ax)
                

        if 'LRCMC' in bounds:
            lrcmc_values_lower, lrcmc_values_upper = (bounds_record['lrcmc_values_lower'], 
                                                      bounds_record['lrcmc_values_upper'])
            plot_bound(empirical_average_range=empirical_average_range,
                       bound_values_lower=lrcmc_values_lower,
                       bound_values_upper=lrcmc_values_upper,
                       title_suffix= 'MCLRA',
                       marker='-',
                       ax=ax)

        if 'LRCMC_corrected' in bounds:
            lrcmc_values_lower, lrcmc_values_upper = (bounds_record['lrcmc_corrected_values_lower'], 
                                                      bounds_record['lrcmc_corrected_values_upper'])
            plot_bound(empirical_average_range=empirical_average_range,
                       bound_values_lower=lrcmc_values_lower,
                       bound_values_upper=lrcmc_values_upper,
                       title_suffix= 'Oneto Bound - Monte Carlo',
                       color='purple',
                       marker='-',
                       ax=ax)

            
        if 'LRCMassart' in bounds:
            lrcmassart_values_lower, lrcmassart_values_upper = (bounds_record['lrcmassart_values_lower'], 
                                                               bounds_record['lrcmassart_values_upper'])
            plot_bound(empirical_average_range=empirical_average_range,
                       bound_values_lower=lrcmassart_values_lower,
                       bound_values_upper=lrcmassart_values_upper,
                       title_suffix= 'Massart LRA',
                       color='orange',
                          marker='--+',
                          ax=ax)
        
        if 'MCRapper' in bounds:            
            mcrapper_epsilon = bounds_record['mcrapper_epsilon']
            plot_bound(empirical_average_range=empirical_average_range,
                       bound_values_lower=empirical_average_range + mcrapper_epsilon,
                       bound_values_upper=empirical_average_range - mcrapper_epsilon,
                       ax=ax,
                       color='green',
                       marker = '--*',
                       title_suffix = 'McRapper')
        
        ax.plot(empirical_average_range,empirical_average_range, label='True Support', color='black')
        ax.legend()


def massart_function(sample_size,number_of_function, max_variance):
    return np.sqrt(2*max_variance*math.log(number_of_function)/sample_size)

def massart_function_variance_shifted(sample_size,number_of_function,max_variance):
    if max_variance < 1/2:         
        shifted_variance = max_variance*(1-max_variance)
    elif max_variance >=  1/2:
        shifted_variance = 1/4
    return massart_function(sample_size=sample_size,
                            number_of_function=number_of_function, 
                            max_variance=shifted_variance)
