import numpy as np
from scipy.optimize import minimize
import autograd.numpy as np
from autograd import grad

alpha = 0.05
T = 50.0
ts = 30.0
ti = 79.0
te = 80.0
e = -2 * np.log(alpha) / T

print("Optimal e: " + str(e))

def asym_prob(k, e):
	p_left_in_bounds = (k**2 / (1 + k**2)) * (1 - np.exp(e*(ts - ti)/k))
	p_right_in_bounds = (1 / (1 + k**2)) * (1 - np.exp(-e * (te - ti) * k))
	return p_left_in_bounds + p_right_in_bounds

def asym_prob_k(k):
	p_left_in_bounds = (k**2 / (1 + k**2)) * (1 - np.exp(e*(ts - ti)/k))
	p_right_in_bounds = (1 / (1 + k**2)) * (1 - np.exp(-e * (te - ti) * k))
	return -1 * (p_left_in_bounds + p_right_in_bounds)

k0 = np.array([1.0])
res = minimize(asym_prob_k, k0, options={'xtol': 1e-8, 'disp': True})

print("Optimal skewness value: " + str(res.x))
print("Asymmetric in range probability: " + str(-1 * asym_prob_k(res.x)))
print("Symmetric in range probability: " + str(-1 * asym_prob_k(np.array([1.0]))))

k = res.x[0]

def asym_prob_e(e):
	p_left_in_bounds = (k**2 / (1 + k**2)) * (1 - np.exp(e*(ts - ti)/k))
	p_right_in_bounds = (1 / (1 + k**2)) * (1 - np.exp(-e * (te - ti) * k))
	return p_left_in_bounds + p_right_in_bounds - (1 - alpha)

def sym_prob_e(e):
	p_left_in_bounds = (1**2 / (1 + 1**2)) * (1 - np.exp(e*(ts - ti)/1))
	p_right_in_bounds = (1 / (1 + 1**2)) * (1 - np.exp(-e * (te - ti) * 1))
	return p_left_in_bounds + p_right_in_bounds - (1 - alpha)

def dx(f, k):
    return abs(f(k))

def newtons_method(f, e0, delta=1e-8):
    diff = dx(f, e0)
    grad_f = grad(f)
    while diff > delta:
        #print("Diff " + str(diff))
        grad_f(e0)
        e0 = e0 - f(e0) / grad_f(e0)
        #print("E " + str(e0))
        diff = dx(f, e0)
    #print('Root is at: ', e0)
    #print('f(e) at root is: ', f(e0))
    return e0


new_asym_e = newtons_method(asym_prob_e, e)
print("Updated asym_e: " + str(new_asym_e))

new_sym_e = newtons_method(sym_prob_e, e)
print("Updated sym_e: " + str(new_sym_e))

print(asym_prob(k, new_asym_e))

print(np.exp((ts - ti)*new_asym_e/k) + np.exp(-k * (te - ti)*new_asym_e) - (1 + k**2)*alpha)


asym_priv_budget_results = []
sym_priv_budget_results = []
ts = 0
for t in range(2, 100):
	T = t
	te = t
	ti = t - 1
	k0 = np.array([1.0])
	res = minimize(asym_prob_k, k0, options={'xtol': 1e-8, 'disp': True})
	k = res.x[0]
	new_asym_e = newtons_method(asym_prob_e, e)
	e = -2 * np.log(alpha) / T
	new_sym_e = newtons_method(sym_prob_e, e)
	e = -2 * np.log(alpha) / T
	asym_priv_budget_results.append(new_asym_e)
	sym_priv_budget_results.append(new_sym_e)

print(asym_priv_budget_results)
print(sym_priv_budget_results)

