from time import time
import math
import pandas as pd
import numpy as np


from scipy.optimize import fixed_point, minimize_scalar
from scipy.special import lambertw

from .mining import transform_to_binary, generate_itemsets, isin_pattern, print_wrt_F

vect_isin = np.vectorize(isin_pattern, signature='(n),(n)->()')



def compute_empirical_support_matrix(data,itemsets):
    return np.array([list(vect_isin(data, item)) for item in itemsets])

def empirical_average(emprical_support_matrix):
    return emprical_support_matrix.sum(axis=1)/emprical_support_matrix.shape[1]


def inverse_poisson_fld_left(a):
    return 1 - np.exp(1+ lambertw((a-1)/np.exp(1), -1).real)

def inverse_poisson_fld_right(a):
    return np.exp(1+ lambertw((a-1)/np.exp(1), 0).real) - 1 

def poisson_fld(x):
    return (1+x)*np.log(1+x) - x


####################
#Counting Arguments:

def sauer_lemma_bound(vcd, sample_size, cf = False):
    if cf:
      #Closed form bound:
      return floor((np.exp(1) * sample_size / vcd) ** vcd)
    else:
        #Binoimal sum bound:
        nf = 0
        for i in range(0, vcd): #Count from 0 to VC(F) - 1, inclusive
            nf += scipy.special.binomial(sample_size, i)
        return nf

def empirical_vc_bound_h_index(itemsets):
  #TODO: Return max d s.t. exist >= d closed itemsets of cardinality d
  pass


def empirical_vc_era_bound(itemsets, max_variance):
    evcd = empirical_vc_bound_h_index(itemsets)
    ni = itemsets.shape[1]
    enf = min(2 ** ni - 1, sauer_lemma_bound(evcd))
    return massart_function(itemsets.shape[0], enf, max_variance)

######################
#ERA Bounds (Massart):
######################


def massart_function_variance_shifted(sample_size,number_of_function,max_variance):
    if max_variance < 1/2:         
        shifted_variance = max_variance*(1-max_variance)
    elif max_variance >=  1/2:
        shifted_variance = 1/4
    return massart_function(sample_size=sample_size,
                            number_of_function=number_of_function, 
                            max_variance=shifted_variance)

def mc_era(empirical_support_matrix, rademacher_variable):
    # Rademacher variable should have shape [sample_size,num_rademacher_trials]
    sample_size = empirical_support_matrix.shape[1]
    num_rademacher_trials = rademacher_variable.shape[-1]    
    # When a is 1-d and b n-dimension, the np.dot occurs on the last axis of a and b.
    # .mean() taken instead of .sum() / num_rademacher_trials
    #return np.abs(np.dot(empirical_support_matrix, rademacher_variable).mean(axis=1))/sample_size
    return np.abs(np.dot(empirical_support_matrix, rademacher_variable))/sample_size



def mc_era_localized(empirical_support_matrix, rademacher_variable, r, sample_size, delta,m=1):
    
    Rn_f_hat_localized = localized_era_mc(empirical_support_matrix,
                                                    rademacher_variable=rademacher_variable, 
                                                    r=r)
    #Rn_f_hat_localized_shifted = Rn_f_hat_localized - min(r, .5)*rademacher_variable.sum()/sample_size
    return np.max(Rn_f_hat_localized,axis=0).mean()

def localized_era_mc(emprical_support_matrix, r, rademacher_variable):
    Pn_f = empirical_average(emprical_support_matrix)
    localization_factor = np.minimum(Pn_f, r)/Pn_f # Alert
    localization_factor = np.nan_to_num(localization_factor, nan=1)
    Rn_f_hat = mc_era(emprical_support_matrix,rademacher_variable)
    return Rn_f_hat * localization_factor[:,np.newaxis]


# def mc_era_localized_corrected(empirical_support_matrix, rademacher_variable, r, sample_size, delta,m=1):
    
#     Rn_f_hat_localized = localized_era_mc(empirical_support_matrix,
#                                                     rademacher_variable=rademacher_variable, 
#                                                     r=r)
#     return np.max(Rn_f_hat_localized,axis=0).mean()

def mc_era_localized_corrected(empirical_support_matrix, rademacher_variable, r, sample_size, delta):
    
    log_delta = log(4/delta)
    num_rademacher_trials = rademacher_variable.shape[-1]
    
    r_hat = 3*r + 5*r*inverse_poisson_fld_right(log_delta/(5*sample_size*r))        
    
    rademacher_monte_carlo_correction = (
                                        2*r_hat*inverse_poisson_fld_right(2*log_delta/(num_rademacher_trials*sample_size*r_hat)) 
                                        + r*inverse_poisson_fld_left(2*log_delta/(sample_size*r))
                                        )
    
    Rn_f_hat_localized = mc_era_localized(empirical_support_matrix, rademacher_variable, r, sample_size, delta,m=1)
    
    #return 2 * np.max(Rn_f_hat_localized,axis=0).mean() + rademacher_monte_carlo_correction
    return 2 * Rn_f_hat_localized + rademacher_monte_carlo_correction


def localized_empirical_rademacher_oneto(psi_era, sample_size,r,delta):
    log_delta = np.log(1/delta)
    localized_empirical_variance = 3*r + 5*r*inverse_poisson_fld_right(log_delta/(5*sample_size*r))
    return psi_era(localized_empirical_variance,delta)#2**d - 1 #Count |F|, ignoring the 0 function



###########################
#Psi Functions:

def psi_ra_oneto(r, sample_size, psi_era,delta):
    log_delta = np.log(1/delta)
    a = 2 * log_delta / (sample_size * r)
    # Correct 2 factor for ERA
    localized_empirical_rademacher_term = localized_empirical_rademacher_oneto(r=r, 
                                                                         sample_size=sample_size,
                                                                         delta=delta,psi_era=psi_era)
    #print('psi_ra_oneto - r=',r)
    return  localized_empirical_rademacher_term + r*inverse_poisson_fld_left(a)

###########################
#Oneto bounds:

def compute_oneto_upper_bound(K,empirical_average, r_U):
    return np.max([empirical_average * K/(K-1), empirical_average + r_U/K])

def compute_oneto_lower_bound(K,empirical_average, r_U):
    #print('########### Compute Oneto Lower bound ')
    #print('Compute Oneto Lower Bound: k=',K)
    #print('Compute Oneto Lower Bound: r_u=',r_U)
    return np.min([empirical_average * K/(K+1), empirical_average - r_U/K])

def initial_guess_for_fixed_point(K, r_hat_star,delta,sample_size):
    initial_guess = r_hat_star * K**2 + (4 * K**2 * np.log(4/delta))/(sample_size)
    #print(initial_guess)
    return initial_guess 

def ru_equation(r, r_hat, delta, K,sample_size):
    # print('ru_equation fixed point: K=',K)
    # print('ru_equation fixed point: r_u_fixed_point_search=',r)
    # print('ru_equation fixed point: r_hat_star=',r_hat)
    # print('\n')
    log_delta = np.log(1/delta)
    
    sqrt_r_times_r_hat = np.sqrt(r * r_hat)
    a1 = 2 * sqrt_r_times_r_hat + r 
    return K*(sqrt_r_times_r_hat + a1* inverse_poisson_fld_right(log_delta/(sample_size*a1)))

def optimize_oneto_bounds(psi_ra_oneto, psi_era, sample_size, delta, emprical_average_range):
    
    def minimize_bound(func_bound, ru_equation, empirical_average, r_hat_star, delta, sample_size, ru_fp_initial_guess, maximum_option=False):
        maximize_constant = -1 if maximum_option else 1
        # TODO Give initial solution K_initial_guess 
        result = minimize_scalar(lambda K : maximize_constant * func_bound(K, empirical_average,
          fixed_point(ru_equation, initial_guess_for_fixed_point(K,r_hat_star, delta, sample_size) *  ru_fp_initial_guess, args=(r_hat_star,delta,K,sample_size),maxiter=10000)),
          bounds = fp_bounds, method="Bounded")
        new_K = result.x
        objective_value = result.fun #Alert +/- 1 
        return new_K, objective_value

    r_hat_star = fixed_point(psi_ra_oneto, 2, args=(sample_size,psi_era,delta))

    
    K_values_lower = []
    K_values_upper = []
    
    obj_values_lower = []
    obj_values_upper = []
    
    
    
    fp_bounds = [1 + 1e-6, 100] #TODO: smart initial guess? Must always be > 1
    ru_fp_initial_guess = 1 #TODO: cached initial guesses?
    for empirical_average in emprical_average_range:
        #print('--------- Empirical Average = {} --------'.format(empirical_average))
    
        new_K_lower, obj_value_lower = minimize_bound(func_bound=compute_oneto_lower_bound,
                                                      ru_equation=ru_equation, 
                                                      empirical_average=empirical_average, 
                                                      r_hat_star=r_hat_star, 
                                                      delta=delta,
                                                      sample_size=sample_size,
                                                      ru_fp_initial_guess=ru_fp_initial_guess,
                                                      maximum_option=True)
        obj_value_lower *= -1
    
    
        new_K_upper, obj_value_upper = minimize_bound(func_bound=compute_oneto_upper_bound,
                                                      ru_equation=ru_equation, 
                                                      empirical_average=empirical_average, 
                                                      r_hat_star=r_hat_star, 
                                                      delta=delta,
                                                      sample_size=sample_size,
                                                      ru_fp_initial_guess=ru_fp_initial_guess,
                                                      maximum_option=False)

        
        obj_values_lower.append(obj_value_lower)
        K_values_lower.append(new_K_lower)
        
        obj_values_upper.append(obj_value_upper)
        K_values_upper.append(new_K_upper)
        
    obj_values_lower = np.array(obj_values_lower)
    obj_values_upper = np.array(obj_values_upper)
    return (obj_values_lower, obj_values_upper, r_hat_star)


###########################
#Union bounds:

def compute_hoeffding_bounds(sample_size,dimension,delta): 
    #TODO: rename Hoeffding union
    
    family_size = (2 ** dimension - 1) #The 0 function does not count
    log_delta = np.log(1/delta)
    return np.sqrt((math.log(2*family_size) + log_delta) / (2*sample_size))

def compute_bennett_bound(empirical_average, sample_size, log_family_size, delta):
    empirical_variance = empirical_average*(1 - empirical_average)
    log_delta = math.log(delta)
    variance_term = np.sqrt(2 * empirical_variance * (np.log(3) + log_family_size - log_delta)/ sample_size )
    return variance_term + 7 * (np.log(3) + log_family_size - log_delta) / (3*sample_size)
