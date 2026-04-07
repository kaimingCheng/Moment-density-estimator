"""
Gneralized Hermite polynomial moment density estimator: kernels, Gram estimators,
MISE bandwidth selection, OSQP positivity post-processing, and moment completion.

Pipeline: empirical moments → pick bandwidth *a* (MISE) → Hermite coefficients →
iterative projection onto nonnegative densities → optional moment completion.
"""

# --- imports ---

import math

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.integrate
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.polynomial.hermite import Hermite
from scipy.special import bernoulli, factorial2, hermite
from scipy.stats import norm

import cvxpy as cp
import osqp
import scipy.sparse as sp

# --- Hermite approximations ---

def hermite_approx(n, x):
  """Evaluate physicist Hermite polynomial H_n(x) (same normalization as scipy.special.hermite)."""
  if n % 2 == 0:
    k = int(n/2)
    v = 0
    for i in range(k+1):
      v += ((2*x)**(2*i))*((-1)**(k-i))/(math.factorial(2*i)*math.factorial(k-i))
  else:
    k = int((n-1)/2)
    v = 0
    for i in range(k+1):
      v += ((2*x)**(2*i+1))*((-1)**(k-i))/(math.factorial(2*i+1)*math.factorial(k-i))
  # print(v)
  return v

def hermite_approx_a(n, x, a=0.1):
  x = int(x/a)
  if n % 2 == 0:
    k = int(n/2)
    v = 0
    for i in range(k+1):
      v += ((2*x)**(2*i))*((-1)**(k-i))/(math.factorial(2*i)*math.factorial(k-i))
  else:
    k = int((n-1)/2)
    v = 0
    for i in range(k+1):
      v += ((2*x)**(2*i+1))*((-1)**(k-i))/(math.factorial(2*i+1)*math.factorial(k-i))
  # print(v)
  return v

# --- Dirac kernel, moments, estimators ---

def dirac(x, a):
  """Gaussian envelope factor N(0, a^2/2) — combined with Hermite terms to build the density estimator."""
  density = norm.pdf(x, loc=0, scale=a / np.sqrt(2))
  return density

def ddirac(x, a, k):
  stn = dirac(x, a)
  if k == 0:
    density = stn*hermite(k)(x/abs(a))
  else:
    norm = (1/abs(a))**k
    # norm = test(a, k)
    # print(norm)
    # density = norm*stn*hermite(k)(x/abs(a))
    density = norm*stn*hermite_approx(k, x/abs(a))
  # print(density)
  return density

def estimator(x, moments, a, n):
  density = 0
  for i in range(n):
    density += ddirac(x, a, i)*moments[i]
  return density

def ddirac_a(x, a, k):
  stn = dirac(x, a)
  if k == 0:
    density = stn*hermite(k)(x/abs(a))
  else:
    norm = (1/abs(a))**k
    # norm = test(a, k)
    # print(norm)
    # density = norm*stn*hermite(k)(x/abs(a))
    density = norm*stn*hermite_approx_a(k, x, a)
  # print(density)
  return density

def estimator_a(x, moments, a, n):
  density = 0
  for i in range(n):
    density += ddirac_a(x, a, i)*moments[i]
  return density

def hermite_sum(x, t, a, n):
  density = 0
  for i in range(n):
    density += ddirac(x, a, i)*(t**i)
  return density

def density(x, t, a):
  density = (1/(np.sqrt(np.pi)*abs(a)))*np.exp(-((t-x)**2)/(a**2))
  return density

def list_moment(n, sigma=1):
  moments = []
  for i in range(n):
    if (i % 2) == 0:
      moments.append(factorial2(i-1)*(sigma**i))
    else:
      moments.append(0)
  moments[0] = 1
  return moments

def list_moment_exp(n, l=1):
  moments = []
  for i in range(n):
    moments.append(math.factorial(i)/(l**i))
  moments[0] = 1
  return moments

def list_moment_logistic(n, b):
  moments = []
  list1 = bernoulli(n)
  for i in range(n):
    if (i % 2) == 0:
      moments.append((2**i-2)*(np.pi**i)*(b**i)*abs(list1[i]))
    else:
      moments.append(0)
  moments[0] = 1
  return moments

def list_moment_logistic_test(n, m):
  moments = []
  list1 = list_moment_logistic(n, np.sqrt(3)/np.pi)
  element = list1[-2] * n
  for i in range(m):
    if i % 2 == 0:
      list1.append(element)
      element *= (n+i+2)
    else:
      list1.append(0)
  return list1


# --- Gram / Hermite coefficient estimators ---

def hermite_kernel(n, x):
  return np.exp((-x**2)/2)*hermite(n)(x)/((2**n)*(math.factorial(n))*(np.sqrt(np.pi)))**0.5

def kernel(x, y, k, fejer=0):
  output = 0
  for j in range(k):
    output += hermite_kernel(j, x)*hermite_kernel(j, y)*(1-j/k)**fejer
  return output

def hermite_coefficient(moments, a, k):
  list1 = []
  for j in range(k):
    d = 0
    norm = (1/2)**j
    for l in range(j+1):
      # density = norm*stn*hermite(k)(x/abs(a))
      if j % 2 == 0:
        if l % 2 ==0:
          c = math.factorial(j)*((2/a)**(l))*((-1)**(int(j/2)-int(l/2)))/(math.factorial(l)*math.factorial(int(j/2)-int(l/2)))
        else:
          c = 0
      else:
        if l % 2 == 0:
          c = 0
        else:
          c = math.factorial(j)*((2/a)**(l))*((-1)**(int((j-1)/2)-int((l-1)/2)))/(math.factorial(l)*math.factorial(int((j-1)/2)-int((l-1)/2)))
      d += c*moments[l]
    list1.append(d*norm/math.factorial(j))

  return list1

def gram_d(x, moments, a, k, fejer=0, method='regular', m=0):
  stn = dirac(x, a)
  density = 0
  f_0 = 0
  g_0 = 0
  for j in range(k):
    d = 0
    norm = (1/2)**j
    for l in range(j+1):
      # density = norm*stn*hermite(k)(x/abs(a))
      if j % 2 == 0:
        if l % 2 ==0:
          c = math.factorial(j)*((2/a)**(l))*((-1)**(int(j/2)-int(l/2)))/(math.factorial(l)*math.factorial(int(j/2)-int(l/2)))
        else:
          c = 0
      else:
        if l % 2 == 0:
          c = 0
        else:
          c = math.factorial(j)*((2/a)**(l))*((-1)**(int((j-1)/2)-int((l-1)/2)))/(math.factorial(l)*math.factorial(int((j-1)/2)-int((l-1)/2)))
      d += c*moments[l]
    if method == 'selective':
      m = k//2
      if j < k//2:
        density += d*norm*hermite_approx(j, x/abs(a))*(1-j/k)**fejer
      else:
        density += d*norm*hermite_approx(j, x/abs(a))*(1-(j-m)/(k-m))**fejer
    elif method == 'two_scale':
      N = k // 2
      N2 = 2*N
      if j < N:
        density += 2*d*norm*hermite_approx(j, x/abs(a))*(1-j/N)**fejer - d*norm*hermite_approx(j, x/abs(a))*(1-j/N2)**fejer
      else:
        density -= d*norm*hermite_approx(j, x/abs(a))*(1-j/N2)**fejer
    elif method == 'one_point':
      f_0 += d*norm*hermite_approx(j, 0)
      g_0 += d*norm*hermite_approx(j, 0)*(1-j/k)**fejer
      density += d*norm*hermite_approx(j, x/abs(a))*(1-j/k)**fejer
    else:
      density += d*norm*hermite_approx(j, x/abs(a))*(1-j/k)**fejer
  if method == 'one_point':
    density = density + (f_0 - g_0)*(kernel(x, 0, k)/kernel(0, 0, k))
  return density*stn

def adaptive_gram_d(x, moments, a, k, fejer=0, method='regular', l2=-3, l1=-2, r1=2, r2=3):
  if x <= l2 or x >= r2:
    return gram_d(x, moments, a, k, fejer=1, method='regular', m=0)
  elif x >= l1 and x <= r1:
    return gram_d(x, moments, a, k, fejer=0, method='regular', m=0)
  elif x >= l2 and x <= l1:
    phi = abs(x-l2)/abs(l2-l1)
    return phi*gram_d(x, moments, a, k, fejer=0, method='regular', m=0) + (1-phi)*gram_d(x, moments, a, k, fejer=1, method='regular', m=0)
  elif x >= r1 and x <= r2:
    phi = abs(x-r2)/abs(r2-r1)
    return phi*gram_d(x, moments, a, k, fejer=0, method='regular', m=0) + (1-phi)*gram_d(x, moments, a, k, fejer=1, method='regular', m=0)

def gram_gamma(x, moments, a, k):
  stn = dirac(x, a)
  density = 0
  for j in range(k):
    d = 0
    norm = (1/2)**j
    for l in range(j+1):
      # density = norm*stn*hermite(k)(x/abs(a))
      if j % 2 == 0:
        if l % 2 ==0:
          c = math.factorial(j)*((2/a)**(l))*((-1)**(int(j/2)-int(l/2)))/(math.factorial(l)*math.factorial(int(j/2)-int(l/2)))
        else:
          c = 0
      else:
        if l % 2 == 0:
          c = 0
        else:
          c = math.factorial(j)*((2/a)**(l))*((-1)**(int((j-1)/2)-int((l-1)/2)))/(math.factorial(l)*math.factorial(int((j-1)/2)-int((l-1)/2)))
      d += c*moments[l]
    density += d*norm*hermite_approx(j, x/abs(a))
  return density*stn

def gram_d2(x, moments, a, k):
  stn = dirac(x, a)
  density = 0
  for j in range(k):
    d = 0
    norm = (1/2)**j
    for l in range(j+1):
      if j % 2 == 0:
        if l % 2 ==0:
          c = math.factorial(j)*((2/a)**(l))*((-1)**(int(j/2)-int(l/2)))/(math.factorial(l)*math.factorial(int(j/2)-int(l/2)))
        else:
          c = 0
      else:
        if l % 2 == 0:
          c = 0
        else:
          c = math.factorial(j)*((2/a)**(l))*((-1)**(int((j-1)/2)-int((l-1)/2)))/(math.factorial(l)*math.factorial(int((j-1)/2)-int((l-1)/2)))
      d += c*moments[l]
    density += d*norm*hermite_approx(j, x/abs(a))
  return density*stn

def estimator_gram(x, moments, a, n):
  stn = dirac(x, a)
  density = 0
  for j in range(n):
    norm = (1/abs(a))**j
    density += norm*moments[j]*hermite_approx(j, x/abs(a))
  return density*stn

def gram_d_a(x, moments, a, k):
  stn = dirac(x, a)
  density = 0
  for j in range(k):
    d = 0
    norm = (1/2)**j
    for l in range(j+1):
      # density = norm*stn*hermite(k)(x/abs(a))
      if j % 2 == 0:
        if l % 2 ==0:
          c = math.factorial(j)*((2/a)**(l))*((-1)**(int(j/2)-int(l/2)))/(math.factorial(l)*math.factorial(int(j/2)-int(l/2)))
        else:
          c = 0
      else:
        if l % 2 == 0:
          c = 0
        else:
          c = math.factorial(j)*((2/a)**(l))*((-1)**(int((j-1)/2)-int((l-1)/2)))/(math.factorial(l)*math.factorial(int((j-1)/2)-int((l-1)/2)))
      d += c*moments[l]
    density += d*norm*hermite_approx(j, x/abs(a))
  return density*stn

def estimator_d(x, moments, a, n, d=0):
  density = 0
  for i in range(d, n):
    c = math.factorial(i)/math.factorial(i-d)
    density += ddirac(x, a, i)*moments[i-d]*c
  return density

def estimator_dk(x, moments, a, n, k, d=0):
  density = 0
  for i in range(d, d+k):
    c = math.factorial(i)/math.factorial(i-d)
    density += ddirac(x, a, i)*moments[i-d]*c
  return density

def estimator_d_test(x, t, a, n, d=0):
  density = 0
  for i in range(d, n):
    c = math.factorial(i)/math.factorial(i-d)
    density += ddirac(x, a, i)*(t**(i-d))*c
  return density

def taylor_coeff(a, k):
  if k == 0:
    return 1
  else:
    c = (((a**2)/2)**(k/2))*(factorial2(k-1)/math.factorial(k))
  return c

def coeff(a, n):
  temp = []
  for i in range(n):
    if i % 2 == 0:
      temp.append(taylor_coeff(a, i))
  temp = np.array(temp)
  matrix_c = np.zeros((len(temp), len(temp)))
  matrix_c[0, :] = temp
  for i in range(1, len(temp)):
    current = 0
    for j in range(i+1):
      current += matrix_c[i-j, j]
    current = -current
    # print(current)
    matrix_c[i, :] = current*temp
  list_c = matrix_c[:, 0]
  return list_c

def temp_list(a, n):
  temp = []
  for i in range(n):
    if i % 2 == 0:
      temp.append(taylor_coeff(a, i))
  temp = np.array(temp)
  return temp

def coeff1(a, n):
  temp = []
  for i in range(n):
    if i % 2 == 0:
      temp.append(taylor_coeff(a[0], i))
  temp = np.array(temp)
  matrix_c = np.zeros((len(temp), len(temp)))
  matrix_c[0, :] = temp
  for i in range(1, len(temp)):
    current = 0
    for j in range(i+1):
      current += matrix_c[i-j, j]
    current = -current
    # print(current)
    matrix_c[i, :] = current*temp_list(a[i], n)
  list_c = matrix_c[:, 0]
  return list_c


def estimator_enhance(x, moments, a, n, early_stop=0):
  list_c = coeff_exact(a, n)
  density = 0
  for k in range(0, (n-early_stop)):
    if k % 2 == 0:
      density += list_c[int(k/2)]*estimator_d(x, moments, a, n, k)
  return density

def estimator_enhance1(x, moments, a, n, early_stop=0):
  list_c = coeff1(a, (n-early_stop))
  density = 0
  for k in range(0, (n-early_stop)):
    if k % 2 == 0:
      density += list_c[int(k/2)]*estimator_d(x, moments, a[int(k/2)], n, k)
  return density

def estimator_enhancek(x, moments, a, n, k=50, early_stop=0):
  list_c = coeff1(a, (n-early_stop))
  density = 0
  for k1 in range(0, (n-early_stop)):
    if k1 % 2 == 0:
      density += list_c[int(k1/2)]*estimator_dk(x, moments, a[int(k1/2)], n, k, k1)
  return density

def estimator_taylor(x, moments, a, n, early_stop=0):
  # list_c = coeff(a, n)
  density = 0
  for k in range(0, (n-early_stop)):
    if k % 2 == 0:
      density += (((-a**2)/2)**(k/2))*estimator_d(x, moments, a, n, k)
  return density

def coeff_exact(a, n):
  list1 = []
  for i in range(n):
    x = (a/2)**(2*i)
    x = x * (-1)**i
    list1.append(x/math.factorial(i))
  return np.array(list1)

# --- Distributions, plotting, QP post-process, pipeline ---

from scipy.stats import norm

class BimodalNormal:
    def __init__(self, mu1=-2, sigma1=1, mu2=2, sigma2=1, w=0.5):
        self.mu1, self.sigma1 = mu1, sigma1
        self.mu2, self.sigma2 = mu2, sigma2
        self.w = w
        # Mock attribute so your pipeline can print the name
        self.dist = type('Mock', (object,), {'name': 'Bimodal Normal'})()

    def pdf(self, x):
        return (self.w * norm.pdf(x, self.mu1, self.sigma1) +
                (1 - self.w) * norm.pdf(x, self.mu2, self.sigma2))

    def rvs(self, size):
        # Determine which component to sample from for each point
        selectors = np.random.rand(size) < self.w
        samples = np.zeros(size)

        # Sample from the first normal
        samples[selectors] = np.random.normal(self.mu1, self.sigma1, size=np.sum(selectors))
        # Sample from the second normal
        samples[~selectors] = np.random.normal(self.mu2, self.sigma2, size=np.sum(~selectors))
        return samples

def emp_moments(dist, n=100000, m=121):
    data = dist.rvs(n) # Shape: (n,)
    powers = np.arange(m) # Shape: (m,)

    # data[:, None] changes shape to (n, 1)
    # Raising to powers creates an (n, m) matrix
    # Taking the mean along axis 0 gives all m moments
    moments = np.mean(data[:, None]**powers, axis=0)
    return moments

# def negative_density_post_process(moments, a=2.5, k=50, dist=norm(0, 1), max_iter=30, termination=-1e-7):
#   n = 0
#   new_coefficients = hermite_coefficient(moments, a, k) # Our initial density estimator
#   roots_current = []
#   while n < max_iter:
#     weights = new_coefficients
#     plot_by_weights(weights, a, dist=dist)
#     local_minima, vals, negative_minima, negative_vals = find_roots(weights, a)
#     print(f'round: {n+1}')
#     if negative_vals == []:
#       print(f'global minimum: {min(vals)}')
#       break
#     else:
#       print(f'global minimum: {min(negative_vals)}')
#       if min(negative_vals) > termination:
#         break
#     roots_current = roots_current + negative_minima
#     print(f'current constraint number: {len(roots_current)}')
#     new_coefficients = gram_d_positive_osqp_root(roots_current, weights, a=a)
#     n+=1
#   if n == max_iter:
#     print('max_iteration_reached')
#   else:
#     print('solution found')
#   return weights

def negative_density_post_process(moments, a=2.5, k=50, dist=None, max_iter=30, termination=-1e-7):
    # Initial state (The "raw" estimate)
    current_weights = hermite_coefficient(moments, a, k)

    # Buffers for backtracking
    # previous_weights will hold the state from "two iterations ago" relative to the failure
    previous_weights = np.copy(current_weights)
    roots_current = []

    for n in range(max_iter):
        # 1. Find negative points for the CURRENT weights
        local_minima, vals, negative_minima, negative_vals = find_roots(current_weights, a)

        print(f'Round: {n+1}')

        if not negative_vals:
            print(f'Global minimum: {min(vals) if len(vals)>0 else "N/A"} (Success)')
            break

        global_min = min(negative_vals)
        if global_min > termination:
            print(f"Termination threshold reached ({global_min:.2e} > {termination}).")
            break

        # 2. Prepare proposed constraints
        # We only commit these to the master list if the QP succeeds
        proposed_roots = roots_current + negative_minima
        # Deduplicate and round to prevent near-singular matrices
        proposed_roots = list(set(np.round(proposed_roots, 6)))

        # 3. Attempt QP Optimization
        # We try to move from current_weights to a better version
        new_weights, success = gram_d_positive_osqp_root(proposed_roots, current_weights, a=a)

        if not success:
            print("!!! QP Solver Failed (Non-convex/Infeasible) !!!")
            print("Backtracking to the weights from TWO iterations ago.")
            # Return the weights from before the iteration that led to this failure
            return previous_weights

        # 4. Success: Move the buffers forward
        # The old 'current' becomes the new 'previous' (2 iterations back relative to next round)
        previous_weights = np.copy(current_weights)
        # Update current to the newly found solution
        current_weights = new_weights
        # Commit the roots
        roots_current = proposed_roots

        print(f'Round {n+1} successful. Constraints: {len(roots_current)}')

    return current_weights

def plot_by_weights(weights, a=2.5, dist=norm(0, 1)):
  #  dist = scipy.stats.logistic(loc=0, scale=(np.sqrt(3)/np.pi))
  #  dist = johnsonsu(1.08, 2.18, loc=1, scale=1.76)
  #  dist = norm(0, 1)
   x = np.arange(-8, 8, 0.1)
   n_points = len(x)
   weights = np.array(weights)  # Target vector
   n = len(weights)

   H = np.zeros((n_points, n))
   for j in range(n):
       H_j = Hermite.basis(j)
       H[:, j] = H_j(x/a)

   true = np.zeros(len(x))
   new = H @ weights
   for i in range(len(x)):
     new[i] = new[i] * dirac(x[i], a=a)
     true[i] = dist.pdf(x[i])

   plt.plot(x, true, label='true')
   plt.plot(x, new, label='appr')
   plt.legend()
   plt.show()
   return None

def plot_by_weights_final(x, weights, a=2.5):
   n_points = len(x)
   weights = np.array(weights)  # Target vector
   n = len(weights)

   H = np.zeros((n_points, n))
   for j in range(n):
       H_j = Hermite.basis(j)
       H[:, j] = H_j(x/a)

   new = H @ weights
   for i in range(len(x)):
     new[i] = new[i] * dirac(x[i], a=a)

   return new

def display(x, weights, moments, dist, a=2.5, k=50, dist_name="Target Distribution"):
  y_true = np.zeros(len(x)) # True density
  est = np.zeros(len(x))

  for t in range(len(x)):
      y_true[t] = dist.pdf(x[t])
      est[t] = gram_d(x[t], moments, a=a, k=k)
  post = plot_by_weights_final(x, weights, a)

  plt.title(f'{dist_name}')
  plt.plot(x, y_true, label='True PDF')
  plt.plot(x, est, label='Raw Estimator')
  plt.plot(x, post, label='Postprocess')
  plt.legend()
  plt.show()

  plt.title(f'{dist_name} log')
  plt.plot(x, np.log(y_true), label='True PDF')
  plt.plot(x, np.log(est), label='Raw Estimator')
  plt.plot(x, np.log(post), label='Postprocess')
  plt.legend()
  plt.show()

  plot_comparison(x, y_true, est, post, a, dist_name)

def display_completed(x, weights_original, known, weights_completed, m, moments, dist, a=2.5, k=50, dist_name="Target Distribution"):
  y_true = np.zeros(len(x)) # True density
  est = np.zeros(len(x))

  for t in range(len(x)):
      y_true[t] = dist.pdf(x[t])
      est[t] = gram_d(x[t], moments, a=a, k=k)
  post_original = plot_by_weights_final(x, weights_original, a)
  post_completed = plot_by_weights_final(x, weights_completed, a)

  # plt.title(f'{dist_name}')
  # plt.plot(x, y_true, label='True PDF')
  # plt.plot(x, est, label='Raw Estimator')
  # plt.plot(x, post, label='Postprocess')
  # plt.legend()
  # plt.show()

  # plt.title(f'{dist_name} log')
  # plt.plot(x, np.log(y_true), label='True PDF')
  # plt.plot(x, np.log(est), label='Raw Estimator')
  # plt.plot(x, np.log(post), label='Postprocess')
  # plt.legend()
  # plt.show()

  plot_comparison_completed(x, y_true, est, post_original, known, post_completed, m, a, dist_name)

def find_roots(weights, a=2.5, region=[-8, 8]):
  H = Hermite(weights)
  f = lambda x: H(x/a)
  Hprime = H.deriv()
  H2prime = Hprime.deriv()

  fprime = lambda x: (1/a) * Hprime(x/a)
  f2prime = lambda x: (1/a**2) * H2prime(x/a)
  # fprime = f.deriv()   # first derivative
  # f2prime = fprime.deriv()  # second derivative (for checking convexity)

  crit_points_y = Hprime.roots()
  crit_points_y = crit_points_y[np.isreal(crit_points_y)].real

  # Map back to x-space
  crit_points = a * crit_points_y
  # crit_points = fprime.roots()  # stationary points (may be complex)
  crit_points = crit_points[np.isreal(crit_points)].real  # keep only real
  crit_points = crit_points[(crit_points >= region[0]) & (crit_points <= region[1])]

  minima = []
  for x in crit_points:
      if f2prime(x) > 0:   # convex (local minimum)
          minima.append(x)
  min_vals = [f(x) for x in minima]
  exact_vals = []
  negative_roots = []
  negative_vals = []
  for i in range(len(min_vals)):
    exact_vals.append(min_vals[i] * dirac(minima[i], a=a))
    if min_vals[i] < 0:
      negative_roots.append(minima[i])
      negative_vals.append(min_vals[i] * dirac(minima[i], a=a))
  return minima, exact_vals, negative_roots, negative_vals

def gram_d_positive_osqp_root(xvals, weights, a=2.5):
  x_vals = np.array(xvals)
  n_points = len(x_vals)
  a1 = np.array(weights)  # Target vector
  n = len(a1)

  # Objective: (1/2)(b - a)^T(b - a)
  # => P = I, q = -a
  P = sp.csc_matrix(np.eye(n))
  q = -a1

  # Constraint: H @ b >= 0
  # Example: b0 - b1 >= 0, b1 - b2 >= 0
  H = np.zeros((n_points, len(weights)))
  for j in range(len(weights)):
      H_j = Hermite.basis(j)
      H[:, j] = H_j(x_vals/a)

  for i in range(n_points):
    coeff = dirac(x_vals[i], a=a)
    H[i, :] = H[i, :] * coeff

  # print(H)

  A = sp.csc_matrix(H)
  l = (-0.0)*np.ones(H.shape[0])   # Lower bounds (H b >= 0)
  u = np.full(H.shape[0], np.inf)  # No upper bound

  try:
      # Setup OSQP problem
      prob = osqp.OSQP()
      prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, scaling=10, max_iter=500000)

      # Solve
      res = prob.solve()

      if res.info.status == 'solved':
          return res.x, True
      else:
          print(f"OSQP Status: {res.info.status}. Skipping update.")
          return weights, False
  except ValueError as e:
      # This catches the 'Problem appears non-convex' error
      print(f"Solver Error encountered: {e}. Returning last valid weights.")
      return weights, False

def plot_comparison(x, y_true, y_raw, y_post, a, dist_name="Target Distribution"):
    """
    Plots the ground truth, the raw Hermite estimator, and the QP-cleaned version.
    dist_name: String for the distribution name (e.g., 'Logistic', 'Student-t').
    """
    # Use the 'ticks' style for a scientific, clean look
    sns.set_style("ticks")
    plt.figure(figsize=(10, 6), dpi=110)

    # 1. Ground Truth (The "Shadow" effect)
    plt.fill_between(x, y_true, color='gray', alpha=0.15, label='True PDF')
    plt.plot(x, y_true, color='gray', linestyle=':', alpha=0.4, linewidth=1)

    # 2. Raw Estimator (High-frequency oscillations)
    plt.plot(x, y_raw, color='#f39c12', alpha=0.6, linewidth=1.5,
             label='Raw Estimator')

    # 3. Post-Processed Estimator (The "Final" product)
    plt.plot(x, y_post, color='#27ae60', linewidth=2.5,
             label='Post-Processed (QP)')

    # --- Title and Labels ---
    plt.title(f"Density Reconstruction: {dist_name}\n"
              f"Scale $a = {a:.3f}$", fontsize=15, fontweight='bold', pad=15)

    plt.xlabel("$x$", fontsize=12)
    plt.ylabel("$f(x)$", fontsize=12)

    # Place legend outside or in a clean spot
    plt.legend(frameon=False, loc='upper right', fontsize=10)

    # Remove top and right spines
    sns.despine()

    plt.tight_layout()
    plt.show()

def plot_comparison_completed(x, y_true, y_raw, post_original, known, post_completed, m, a, dist_name="Target Distribution"):
    """
    Plots the ground truth, the raw Hermite estimator, and the QP-cleaned version.
    dist_name: String for the distribution name (e.g., 'Logistic', 'Student-t').
    """
    # Use the 'ticks' style for a scientific, clean look
    sns.set_style("ticks")
    plt.figure(figsize=(10, 6), dpi=110)

    # 1. Ground Truth (The "Shadow" effect)
    plt.fill_between(x, y_true, color='gray', alpha=0.15, label='True PDF')
    plt.plot(x, y_true, color='gray', linestyle=':', alpha=0.4, linewidth=1)

    # # 2. Raw Estimator (High-frequency oscillations)
    # plt.plot(x, y_raw, color='#f39c12', alpha=0.6, linewidth=1.5,
    #          label='Raw Estimator')

    # 3. Post-Processed Original Estimator (The "Final" product)
    plt.plot(x, post_original, color='#f39c12', linewidth=2.5,
             label=f'Estimator with original {known} moments')

    # 4. Post-Processed Estimator (The "Final" product)
    plt.plot(x, post_completed, color='#27ae60', linewidth=2.5,
             label=f'Estimator with completed {m} moments')

    # --- Title and Labels ---
    plt.title(f"Density Reconstruction with completed moments: {dist_name}\n"
              f"Scale $a = {a:.3f}$", fontsize=15, fontweight='bold', pad=15)

    plt.xlabel("$x$", fontsize=12)
    plt.ylabel("$f(x)$", fontsize=12)

    # Place legend outside or in a clean spot
    plt.legend(frameon=False, loc='upper right', fontsize=10)

    # Remove top and right spines
    sns.despine()

    plt.tight_layout()
    plt.show()

def run_hermite_estimation_pipeline(dist, n=100000, m=60, M=120, a_grid = np.linspace(1.5, 4.0, 200), x_range = np.linspace(-10, 10, 200), moments=None, dist_name='Target Distribution', richardson=False, known=0):
    """
    Complete pipeline: 1. Generate moments, 2. Optimize scale 'a',
    3. Post-process weights, 4. Visualize.
    """
    print(f"Starting pipeline for {dist.dist.name} distribution...")

    # --- Step 1: Generate Empirical Moments ---
    if moments is None:
        print(f"Generating {n} samples and empirical moments for {dist.dist.name}...")
        data = dist.rvs(n)
        powers = np.arange(M + 1)
        # Vectorized calculation
        moments1 = np.mean(data[:, None]**powers, axis=0)
    else:
        print(f"Using provided moments for {dist.dist.name} pipeline...")
        moments1 = np.array(moments)
        if len(moments1) < M + 1:
            raise ValueError(f"Provided moments array must have length at least {M + 1}")

    # --- Step 2: Optimized MISE for Best 'a' ---
    # Define a grid for 'a' search
    if moments is None:
      # m_min = min(35, m)
      # M_min = min(70, M)
      m_min = m
      M_min = M
    else:
      m_min = min(35, int(0.5*known))
      M_min = min(70, known)
    # m_min = m
    # M_min = M

    best_a = pick_best_a(moments1, a_grid, n=n, m=m_min, M=M_min, richardson=richardson)
    print(f"Optimal scale 'a' found: {best_a:.4f}")

    # --- Step 3: Apply Post-processing (QP/Negative Density Correction) ---
    # We use the optimized 'a' here
    weights = negative_density_post_process(moments1[:m], k=m, a=best_a, dist=dist)

    # --- Step 4: Display/Visualize ---

    display(x_range, weights, moments1, dist, a=best_a, k=m, dist_name=dist_name)
    plt.show()

    return best_a, weights

def plot_completed_vs_original(dist, n=100000, m=60, M=120, a_grid = np.linspace(1.5, 4.0, 200), x_range = np.linspace(-10, 10, 200), moments=None, dist_name='Target Distribution', richardson=False, known=0):
    """
    Complete pipeline: 1. Generate moments, 2. Optimize scale 'a',
    3. Post-process weights, 4. Visualize.
    """
    print(f"Starting pipeline for {dist.dist.name} distribution...")

    # --- Step 1: Generate Empirical Moments ---
    if moments is None:
        print(f"Generating {n} samples and empirical moments for {dist.dist.name}...")
        data = dist.rvs(n)
        powers = np.arange(M + 1)
        # Vectorized calculation
        moments1 = np.mean(data[:, None]**powers, axis=0)
    else:
        print(f"Using provided moments for {dist.dist.name} pipeline...")
        moments1 = np.array(moments)
        # moments2 = np.array(moments[:m])
        if len(moments1) < M + 1:
            raise ValueError(f"Provided moments array must have length at least {M + 1}")

    # --- Step 2: Optimized MISE for Best 'a' ---
    # Define a grid for 'a' search
    if moments is None:
      # m_min = min(35, m)
      # M_min = min(70, M)
      m_min = m
      M_min = M
    else:
      m_min = min(35, int(0.5*known))
      M_min = min(70, known)
    # m_min = m
    # M_min = M

    best_a = pick_best_a(moments1, a_grid, n=n, m=m_min, M=M_min, richardson=richardson)
    print(f"Optimal scale 'a' found: {best_a:.4f}")

    # --- Step 3: Apply Post-processing (QP/Negative Density Correction) ---
    # We use the optimized 'a' here
    weights_original = negative_density_post_process(moments1[:known], k=known, a=best_a, dist=dist)
    weights_completed = negative_density_post_process(moments1[:m], k=m, a=best_a, dist=dist)

    # --- Step 4: Display/Visualize ---

    display_completed(x_range, weights_original, known, weights_completed, m, moments1, dist, a=best_a, k=m, dist_name=dist_name)
    plt.show()

    return best_a, weights_original, weights_completed

# --- Bandwidth (MISE), pick_best_a, Richardson ---

from numpy.polynomial.hermite import hermval


def plot_balanced_mise(a_grid, mise_list, best_a, height_factor=5):
    """
    Plots a focused, balanced region of the MISE curve.
    height_factor: How many times the minimum MISE the endpoints should reach.
    """
    a_grid = np.asarray(a_grid)
    mise_list = np.asarray(mise_list)
    min_err = np.min(mise_list)
    target_err = min_err * height_factor

    # Find the index of the minimum
    idx_min = np.argmin(mise_list)

    # Search left from the minimum for the first point above target_err
    left_side = mise_list[:idx_min]
    left_idx = np.where(left_side >= target_err)[0]
    start_idx = left_idx[-1] if len(left_idx) > 0 else 0

    # Search right from the minimum for the first point above target_err
    right_side = mise_list[idx_min:]
    right_idx = np.where(right_side >= target_err)[0]
    end_idx = (idx_min + right_idx[0]) if len(right_idx) > 0 else len(a_grid) - 1

    # Slice the data
    a_zoom = a_grid[start_idx:end_idx+1]
    mise_zoom = mise_list[start_idx:end_idx+1]

    # --- Plotting ---
    sns.set_theme(style="white", context="talk")
    plt.figure(figsize=(9, 6), dpi=110)

    # Main Line
    plt.plot(a_zoom, mise_zoom, color='#2c3e50', linewidth=3, label=r'$\overline{\text{MISE}}$')

    # Aesthetic Fill
    plt.fill_between(a_zoom, mise_zoom, min_err, color='#3498db', alpha=0.1)

    # Highlight Optimal a*
    plt.axvline(best_a, color='#e74c3c', linestyle=':', linewidth=2, alpha=0.6)
    plt.scatter(best_a, min_err, color='#e74c3c', s=100, zorder=5,
                label=f'Optimal $a^* = {best_a:.3f}$')

    # Clean Up
    plt.yscale('log')
    plt.xlabel(r"Bandwidth Parameter $a$", fontsize=14)
    plt.ylabel(r"Error Magnitude $\overline{\text{MISE}}$", fontsize=14)
    plt.title("Balanced MISE Optimization Window", pad=20, fontweight='bold')

    # Remove top/right spines for a modern look
    sns.despine(trim=True)
    plt.legend(frameon=False, loc='upper center')

    plt.tight_layout()
    plt.show()

def fancy_mise_plot(a_grid, mise_list, best_a, threshold_factor=100):
    """
    Plots MISE with LaTeX formatting and adaptive range cropping.
    threshold_factor: Crops the plot to the region where MISE < min_mise * threshold_factor.
    """
    # 1. Ensure inputs are numpy arrays
    a_grid = np.asarray(a_grid)
    mise_list = np.asarray(mise_list)
    min_mise = np.min(mise_list)

    # 2. Adaptive Cropping Logic
    # We only want to plot the region where the error hasn't "exploded"
    mask = mise_list <= (min_mise * threshold_factor)

    # To keep the plot continuous, we find the first and last index of the valid mask
    valid_indices = np.where(mask)[0]
    if len(valid_indices) > 0:
        start, end = valid_indices[0], valid_indices[-1]
        # Add a little buffer if possible
        start = max(0, start - 2)
        end = min(len(a_grid) - 1, end + 2)

        a_plot = a_grid[start:end+1]
        mise_plot = mise_list[start:end+1]
    else:
        a_plot, mise_plot = a_grid, mise_list

    # 3. Styling
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    # Use the upper bound symbol as requested in the legend
    ax.plot(a_plot, mise_plot, color='#2c3e50', linewidth=2.5, label=r'$\overline{\text{MISE}}$')
    ax.fill_between(a_plot, mise_plot, alpha=0.1, color='#3498db')

    # 4. Highlight Optimal a*
    ax.axvline(best_a, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.scatter(best_a, min_mise, color='#e74c3c', s=80, zorder=5,
               label=f'Optimal $a^* = {best_a:.4f}$')

    # Formatting
    ax.set_yscale('log')
    ax.set_xlabel("Bandwidth Parameter ($a$)", fontsize=12)
    ax.set_ylabel(r"Error Magnitude ($\overline{\text{MISE}}$)", fontsize=12)
    ax.set_title("Adaptive MISE Optimization Window", fontsize=14, fontweight='bold')

    # 5. Legend and Aesthetics
    ax.legend(frameon=True, shadow=True, loc='upper right')
    sns.despine()

    # --- INSET PLOT: Super-Zoom on the actual curve bottom ---
    axins = inset_axes(ax, width="30%", height="30%", loc='lower center', borderpad=3)

    # Zoom range: very tight around the minimum
    zoom_mask = (a_grid > best_a - 0.1) & (a_grid < best_a + 0.1)
    if np.any(zoom_mask):
        axins.plot(a_grid[zoom_mask], mise_list[zoom_mask], color='#2c3e50', lw=2)
        axins.axvline(best_a, color='#e74c3c', ls='--')
        axins.set_title("Zoom at $a^*$", fontsize=9)

    axins.tick_params(labelsize=8)

    plt.show()

def hermite_H(n, x):
    """Compute physicists' Hermite polynomial H_n(x)."""
    coeffs = np.zeros(n + 1)
    coeffs[n] = 1.0
    return hermval(x, coeffs)

def ua(x, a):
    """Gaussian envelope."""
    return np.exp(-(x / a) ** 2) / (a * np.sqrt(np.pi))

def delta_ij(i, j, a):
    """Coefficient δ_{i,j} of x^j in physicists' Hermite polynomial H_i(x/a)."""
    if (i - j) % 2 != 0 or j > i:
        return 0.0
    r = (i - j) // 2
    return ((-1)**r) * ((2/a)**j) * math.factorial(i) / (math.factorial(r) * math.factorial(j))

def delta_scaled(k, i, a):
    if (k - i) % 2 != 0 or i > k:
        return 0.0
    r = (k - i) // 2
    return ((-1)**r) * ((2/a)**i) / (2**k * math.factorial(r) * math.factorial(i))

def delta_matrix(max_order):
    """Return (max_order+1)x(max_order+1) matrix of δ_{i,j}."""
    δ = np.zeros((max_order + 1, max_order + 1))
    for i in range(max_order + 1):
        for j in range(i + 1):
            δ[i, j] = delta_ij(i, j)
    return δ


def mse_estimator(x, mu, a, n, m, M):
    """
    Compute MSE(x) for the Hermite density estimator given:
    - x: scalar or array
    - mu: array of population moments [μ_0, μ_1, ..., μ_{M+m}]
    - delta: (M+1)×(M+1) matrix of δ_{j,i}
    - a: scale parameter
    - n: sample size
    - m: truncation order for variance part
    - M: maximum order for bias part
    """
    x = np.atleast_1d(x)
    ua_x = ua(x, a)

    # ---- 1. Covariance between empirical moments ----
    def cov_mu(i, j):
        return (mu[i + j] - mu[i] * mu[j]) / n

    # def cov_mu(i, j):
        # return np.cov(mu**i, mu**j, bias=True)[0,1]

    # ---- 2. Variance term ----
    var_term = np.zeros_like(x)
    for i in range(m + 1):
        for j in range(m + 1):
            cov_ij = cov_mu(i, j)
            # compute F_i(x) and F_j(x)
            Fi = np.zeros_like(x)
            Fj = np.zeros_like(x)
            for k in range(i, m + 1):
                # Fi += delta_ij(k, i, a) * hermite_H(k, x / a) / (2 ** k * math.factorial(k))
                Fi += delta_scaled(k, i, a) * hermite_H(k, x / a)
            for k in range(j, m + 1):
                # Fj += delta_ij(k, j, a) * hermite_H(k, x / a) / (2 ** k * math.factorial(k))
                Fi += delta_scaled(k, j, a) * hermite_H(k, x / a)
            var_term += cov_ij * Fi * Fj

    # ---- 3. Bias term ----
    bias_term = np.zeros_like(x)
    for j in range(m + 1, M + 1):
        coeff = 0
        for i in range(j + 1):
            coeff += delta_ij(j, i, a) * mu[i] / (2 ** j * math.factorial(j))
        bias_term += coeff * hermite_H(j, x / a)
    bias_sq = bias_term ** 2

    # ---- 4. Combine ----
    mse_x = ua_x ** 2 * (var_term + bias_sq)

    return mse_x


def mise_estimator(mu, a, n, m, M):
    def cov_mu(i, j):
        return (mu[i + j] - mu[i] * mu[j]) / n

    # ---- 2. Variance term ----
    var_term = 0
    for i in range(m + 1):
        for j in range(m + 1):
            cov_ij = cov_mu(i, j)
            Fi = 0
            for k in range(max(i,j), m + 1):
                Fi += delta_ij(k, i, a) * delta_ij(k, j, a) / (2 ** k * math.factorial(k))
            var_term += cov_ij * Fi
    # var_term = get_variance(m, mu, n, a)
    # var_term = get_variance_final_fix(m, mu, n, a)

    # ---- 3. Bias term ----
    bias_term = 0
    for j in range(m + 1, M + 1):
        coeff = 0
        for i in range(j + 1):
            coeff += delta_ij(j, i, a) * mu[i]
        bias_term += (coeff ** 2) / (2 ** j * math.factorial(j))

    # ---- 4. Combine ----
    # print(var_term)
    # print(bias_term)
    mise = var_term + bias_term
    # print(f'var: {var_term}')
    # print(f'bias: {bias_term}')

    return mise


def pick_best_a(moments, a_grid, n, m, M, richardson=False):
  best_a = a_grid[0]
  mise_list = []
  best_mise = np.inf
  for i in a_grid:
    if richardson == False:
      current = mise_estimator(moments, i, n, m, M)
    else:
      current = richardson_limit_estimator(moments, i, n, m, M_base=30)
    mise_list.append(current)
    if current < best_mise:
      best_a = i
      best_mise = current

  plt.plot(a_grid, mise_list)
  plt.xlabel("bandwidth a")
  plt.ylabel("MISE(a)")
  plt.title("MISE of Hermite Estimator vs. bandwidth a")
  plt.show()
  fancy_mise_plot(a_grid, mise_list, best_a)
  # plot_balanced_mise(a_grid, mise_list, best_a)
  print(best_mise)

  return best_a




def richardson_limit_estimator(mu, a, n, m, M_base, k=None):
    """
    Extrapolates the MISE as M -> infinity.
    M_base: The starting M value.
    k: The order of decay. If None, it will be estimated.
    """
    # 1. Compute three points to check convergence behavior
    S1 = mise_estimator(mu, a, n, m, M_base)
    S2 = mise_estimator(mu, a, n, m, M_base * 2)
    S3 = mise_estimator(mu, a, n, m, M_base * 4)

    # 2. Estimate k (the order of the tail) if not provided
    # Ratio of differences R = (S2 - S1) / (S3 - S2)
    # k = log2(R)
    if k is None:
        ratio = (S1 - S2) / (S2 - S3)
        k = np.log2(abs(ratio))
        print(f"Detected error decay order k ≈ {k:.2f}")

    # 3. Apply Richardson Extrapolation Formula
    # S_infinity ≈ S_fine + (S_fine - S_coarse) / (2^k - 1)
    refined_val = S3 + (S3 - S2) / (2**k - 1)

    print(f"Original (M={M_base*4}): {S3:.10f}")
    print(f"Extrapolated (M=∞): {refined_val:.10f}")

    return refined_val

# --- Moment completion ---

def iterative_moment_completion(initial_moments, target_order, a=None):
    """
    Iteratively completes a moment sequence by setting w_j = 0 for all j > m.
    """
    moments = list(initial_moments)
    m_start = len(moments)

    # Estimate scale 'a' if not provided
    if a is None:
        variance = moments[2] - moments[1]**2
        a = np.sqrt(2 * variance)

    # Iteratively find the next moment mu_j that makes w_j = 0
    for j in range(m_start, target_order + 1):
        H_poly = hermite(j)
        coeffs = H_poly.coeffs # [alpha_j, alpha_{j-1}, ..., alpha_0]

        alpha_leading = coeffs[0]
        known_sum = 0
        for i, alpha_k in enumerate(coeffs[1:]):
            k = j - (i + 1)
            known_sum += alpha_k * (moments[k] / (a**k))

        mu_next = - (known_sum * (a**j)) / alpha_leading
        moments.append(mu_next)

    return np.array(moments), a

def calculate_energy_sum_list(moments, a):
    """
    Calculates the cumulative energy sum E_i = sum_{j=0}^i w_j^2
    """
    M = len(moments)
    weights = []

    for j in range(M):
        # Normalization factor for orthonormal Hermite basis
        norm_factor = np.sqrt(a * (2**j) * math.factorial(j) * np.sqrt(np.pi))

        H_poly = hermite(j)
        w_j = 0
        for i, alpha_k in enumerate(H_poly.coeffs):
            k = j - i
            w_j += alpha_k * (moments[k] / (a**k))

        w_j /= norm_factor
        weights.append(w_j**2)

    return np.cumsum(weights)

def plot_energy_sum_comparison(moments1, moments2, a, m, labels=["Distribution A", "Distribution B"]):
    """
    Calculates and plots the cumulative energy sums for two moment sequences.

    Args:
        moments1: First moment sequence (array-like)
        moments2: Second moment sequence (array-like)
        a: The bandwidth parameter used in the Hermite expansion
        labels: List of labels for the legend
    """
    # Calculate the energy sums using your existing function
    energy_sum1 = np.log(calculate_energy_sum_list(moments1, a)[:m])
    energy_sum2 = np.log(calculate_energy_sum_list(moments2, a)[:m])

    # Create the x-axis (Truncation Order M)
    orders = np.arange(len(energy_sum1))

    plt.figure(figsize=(9, 5))

    # Plot both sequences
    plt.plot(orders, energy_sum1, label=labels[0], color='#1f77b4', marker='o', markersize=4, linewidth=1.5)
    plt.plot(orders, energy_sum2, label=labels[1], color='#ff7f0e', marker='s', markersize=4, linewidth=1.5)

    # Adding plot details
    plt.xlabel('Truncated Order ($M$)', fontsize=11)
    plt.ylabel(r'Cumulative Energy $E_M = \sum_{j=0}^M w_j^2$', fontsize=11)
    plt.title('Spectral Energy Convergence Comparison', fontsize=13, fontweight='bold')

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='best', frameon=True)

    # Optional: If the energy grows exponentially, a log scale is often better
    # plt.yscale('log')

    plt.tight_layout()
    plt.show()

def calculate_energy_sum(moments, m, a):
    """Calculates the sum of squares of orthonormal Hermite coefficients w_j."""
    w_vector = []
    for j in range(m + 1):
        H_poly = hermite(j)
        coeffs = H_poly.coeffs
        expected_val_H = 0
        degree = len(coeffs) - 1
        for idx, alpha in enumerate(coeffs):
            k = degree - idx
            # E[ (x/a)^k ] = mu_k / (a^k)
            expected_val_H += alpha * (moments[k] / (a**k))

        # Orthonormalization: w_j = c_j / sqrt(2^j * j!)
        norm_factor = np.sqrt((2**j) * factorial(j))
        w_vector.append(expected_val_H / norm_factor)

    return np.sum(np.array(w_vector)**2)
