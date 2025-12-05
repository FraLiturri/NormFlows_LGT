import numpy as np
import torch
from scipy.special import logsumexp as logsumexp_np
from neumc.utils import ess_lw


def normalize_logw(logw): #! where is this used?;
    return logw - torch.logsumexp(logw, dim=0) + np.log(len(logw)) 
    #logsumexp(x_j) = log(sum_i(exp(x_i))) -> softmax in log space (dim = 0 implies scalar output); 

def nis(o_samples, logq, logp): #neural importance sampling; 
    logw = logp - logq
    logw = logw - logw.max() #for numerical stability: subtracting max value prevents overflow or underflow when exponentiating; 
                             #furthermore, it preservers the softmax property (constant shift); 

    w = torch.exp(logw)
    w = w / torch.exp(torch.logsumexp(logw, dim=0)) #w is normalized (discrete formula); 

    return (w * o_samples).sum() #since w is already normalized, we can sample <O> directly (discrete formula);


def nis_lw_np(o_samples, logw): #equivalent to the function before but using numpy arrays;
    logw = logw - logw.max()

    w = np.exp(logw)
    w = w / np.exp(logsumexp_np(logw, axis=0)) #logsumexp_np is the numpy equivalent of torch.logsumexp
                                               #useful when working with numpy arrays;

    return (w * o_samples).sum()


def nis_np(o_samples, logq, logp):    #neural importance sampling using numpy arrays:
    logw = logp - logq                #in this case the inputs are logq and logp (as numpy), instead of a single array logw; 
    return nis_lw_np(o_samples, logw) #calling the previous function;


def cov_1_2_p(x, y, lw):
    '''Computes the expected value of (X - E[X]) * (Y - E[Y])^2, 
    where the expectation is taken with respect to the normalized weights [w = exp(lw)].'''
    dx = x - x.mean()
    dy = y - y.mean()
    return nis_lw_np(dx * dy * dy, lw)


def nis_error_np(o, lw): #computes the error of the NIS estimate of <O> given samples "o" and log-weights "lw"; 
    n = len(o)
    o_bar = nis_lw_np(o, lw) #expected value of O; 
    o2_bar = nis_lw_np(o * o, lw) #expected value of O^2;
    var_o = o2_bar - o_bar**2 #variance of O (\sigma^2_O);
    t1 = var_o / ess_lw(lw) #! ??? 
    w = np.exp(lw) 
    t2 = cov_1_2_p(w, o, lw) / w.mean()
    return np.sqrt((t1 + t2) / n), t1, t2


def nis_lw_with_err_np(o_samples, logw):
    o = nis_lw_np(o_samples, logw)
    o2 = nis_lw_np(o_samples**2, logw)

    err, t1, t2 = nis_error_np(o_samples, logw)
    return o, err, t2 / t1
