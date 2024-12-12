# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:50:22 2024

@author: illsl
"""

import pickle
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Load data
with open('simulation_results.pkl', 'rb') as f:
    data = pickle.load(f)

average_tracker = data['average_tracker']
end_prop_tracker = data['end_prop_tracker']
std_tracker = data['std_tracker']


def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    return (A1 * np.exp(-((x - mu1)**2) / (2 * sigma1**2)) +
            A2 * np.exp(-((x - mu2)**2) / (2 * sigma2**2)))

def gaussian(x,A,mu,sigma):
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

# Fit the data
def fit_double_gaussian(data, bins):
    
    counts, bin_edges = np.histogram(data, bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers
    
    
    p0 = [30, 0.55, 0.02, 25, 0.83, 0.07]
    
    
    bin_errors = np.zeros(len(bin_centers))
    for i in range(len(bin_centers)):
        bin_error = []
        # Find the data points within the current bin
        in_bin = np.where((data >= bin_edges[i]) & (data < bin_edges[i + 1]))[0]
        
        
        if np.any(in_bin):
            for x in in_bin:
                bin_error.append(std_tracker[x] ** 2)   
            bin_errors[i] = np.sqrt(np.sum(bin_error))
        else:
            bin_errors[i] = 0.1  
            
    # Error due to counts
    bin_errors_counts = np.zeros_like(counts)
    non_zero_counts = counts > 0
    bin_errors_counts[non_zero_counts] = np.sqrt(counts[non_zero_counts])

    # Combine errors quadratically
    bin_errors_total = np.sqrt(bin_errors**2 + bin_errors_counts**2)

    # Exclude zero counts from the fitting process
    valid = counts > 0
    valid_bin_centers = bin_centers[valid]
    valid_counts = counts[valid]
    valid_errors = bin_errors_total[valid]
    
    
    # Fit the curve
    params, covariance = curve_fit(double_gaussian, valid_bin_centers, valid_counts, p0=p0, sigma=valid_errors)

    # Calculate standard errors from the diagonal of the covariance matrix
    param_errors = np.sqrt(np.diag(covariance))

    return params, valid_bin_centers, valid_counts, param_errors, valid_errors


def plot_gaussian_seperatly(data, params, bin_centers, counts, errors,bins): 
    
    x = np.linspace(min(bin_centers), max(bin_centers), 1000)
    gauss_1  = gaussian(x, params[0],params[1],params[2])
    gauss_2  = gaussian(x, params[3],params[4],params[5])
    
    plt.hist(data, bins=bins, density=False, alpha=0.6, color='gray', label='Histogram')
    plt.plot(x,gauss_1)
    plt.plot(x,gauss_2)
    plt.ylabel('Counts')
    plt.xlabel('proportion of good bacteria')
    plt.tight_layout()
    plt.savefig(fname = 'gaussians_separetly.png', dpi = 1000)
    plt.show()
    
    
def chi_squared_calc(bin_centers, counts, errors, params):
    
    y_fit = double_gaussian(bin_centers, *params)
    
    residuals = counts - y_fit
    
    chi_squared = np.sum((residuals / errors) ** 2)
    
    # Degrees of freedom: number of data points - number of parameters
    dof = len(counts) - len(params)
    reduced_chi_squared = chi_squared / dof if dof > 0 else np.inf  
    return chi_squared,reduced_chi_squared,residuals



def plot_double_gaussian_fit_with_histogram(data, params, bin_centers, counts, errors, bins,residuals):
    
    x = np.linspace(min(bin_centers), max(bin_centers), 1000)
    fit_curve = double_gaussian(x, *params)
    
    fig, axs = plt.subplots(2, 1, figsize=[8,4], gridspec_kw={'height_ratios': [3, 1]},sharex=True)
    
   
    axs[0].hist(data, bins=bins, density=False, alpha=0.6, color='gray', label='Histogram')
    axs[0].errorbar(bin_centers, counts, yerr=errors, fmt='x', color='black',markersize = 8, label='Data with Errors')
    axs[0].plot(x, fit_curve, 'r-', label='Double Gaussian Fit')
   
    axs[0].set_xlabel('Proportion of good bacteria')
    axs[0].set_ylabel('Counts')
    axs[0].legend()

    
    axs[1].errorbar(bin_centers, residuals, yerr=errors, fmt='o', color='black', label='Residuals')
    axs[1].axhline(0, color='red', linestyle='--', linewidth=1)
    
    axs[1].set_xlabel('Proportion of good bacteria')
    axs[1].set_ylabel('Residuals')
    axs[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig(fname = 'curve_fit_with_residuals', dpi = 1000)
   
    plt.show()



def run_analysis(data, bins=10):
    params, bin_centers, counts, param_errors, errors = fit_double_gaussian(data, bins)
    chi_squared,reduced_chi_squared,residuals = chi_squared_calc(bin_centers, counts, errors, params)
    plot_double_gaussian_fit_with_histogram(data, params, bin_centers, counts, errors, bins,residuals)
    plot_gaussian_seperatly(data, params, bin_centers, counts, errors, bins)
    
    # Print the fitted parameters with their errors
    print("Fitted Parameters with Errors:")
    print(f"A1: {params[0]:.2f} ± {param_errors[0]:.3f}")
    print(f"mu1: {params[1]:.2f} ± {param_errors[1]:.3f}")
    print(f"sigma1: {params[2]:.2f} ± {param_errors[2]:.3f}")
    print(f"A2: {params[3]:.2f} ± {param_errors[3]:.3f}")
    print(f"mu2: {params[4]:.2f} ± {param_errors[4]:.3f}")
    print(f"sigma2: {params[5]:.2f} ± {param_errors[5]:.3f}")
    
    print(f"chi squared:{chi_squared:.2f}")
    print(f"reduced chi squared:{reduced_chi_squared:.2f}")


number_of_bins = 30
run_analysis(end_prop_tracker, bins=number_of_bins)
