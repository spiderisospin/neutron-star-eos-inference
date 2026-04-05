#!/usr/bin/env python
# coding: utf-8

# hey, im creating the data generator part here. The function comments are still kind of vague, but thats because im trying to make a general sketch of whats going on, before polishing it. let me know if u have improvements

# In[4]:


from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import root
import random
from scipy.stats import norm, uniform
from PIL import Image
import statistics
import os
import time


# In[5]:


#constants
c_light = 2.997*1e10
hbar = 6.582*1e-22
MeV = 1.602*1e-6
Kp = MeV / (hbar**3 * c_light**3)
Krho = MeV/(hbar**3 * c_light**5)
fm = (1e-13)**(-3)
n0 = 0.16 * fm
m = 1.675 * 1e-24
M_sun = 1.988 * 1e33
G_const = 6.67*1e-11*1e6/1e3
Lu = (1 / M_sun) * (c_light ** 2) * (1/G_const)
Pu = (1/M_sun) / (Lu ** 3) / (c_light **2)
rhou = (1/ M_sun ) / (Lu **3)
KK = ((2.997*1e5)**2)/(6.67*1e-20 * 1.988 *1e30)


# In[6]:


rho_t = 2*n0*m*rhou
rho_fin = 12*n0*m*rhou
pc = 200**4 * Kp * Pu
r0 = 1e-5
a0=1
f0=1
h0=1
H0 = a0 * r0**2
beta0 = 2 * a0 * r0
rspan = (r0, 200)
Lamb_arr = [-(194.0) * Kp * Pu, -(150.0) * Kp * Pu, -(120.0) * Kp * Pu, -(95.0) * Kp * Pu,
    -(50.0) * Kp * Pu, 0, (50.0)
            * Kp * Pu, (95.0) * Kp * Pu, (120.0) * Kp * Pu,
    (194.0) * Kp * Pu]

M_norm = 3
R_norm = 20


# In[7]:


ap4 = pd.read_csv("Rescaledap4.dat", sep = r'\s+', header=None)
sly = pd.read_csv("Rescaledsly.dat", sep = r'\s+', header=None)
ap4.columns=['p', 'eps', 'rho']
sly.columns=['p', 'eps', 'rho']

#training data
rho_train = pd.read_csv("matrixrho2.dat", sep = r'\s+', header=None)
cs_train = pd.read_csv("matrixcs2.dat", sep = r'\s+', header=None)
rho_train.columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7']
cs_train.columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7']

#test data
rho_test = pd.read_csv("matrixrhoTest.dat", sep = r'\s+', header=None)
cs_test = pd.read_csv("matrixcsTest.dat", sep = r'\s+', header=None)
rho_test.columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7']
cs_test.columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7']

sly = sly.values
ap4 = ap4.values

rho_train = rho_train.values
cs_train = cs_train.values
rho_test = rho_test.values
cs_test = cs_test.values


# In[8]:


def find_pos(matrix, val, col):
  """ """
  col_data = matrix[:, col]
  for i in range(len(col_data) - 1):
    dx1 = abs(col_data[i] - val)
    dx2 = abs(col_data[i+1] - val)

    if col_data[i+1] >= val:
      if dx1<dx2:
        return i
      else:
        return i+1

def deriv(eos, max_idx):
  """s = dp/deps up to max_idx """
  s = 0
  for i in range(max_idx):
    dp = eos[i+1,0] - eos[i,0]
    deps = eos[i+1,1] - eos[i,1]
    s = dp / deps
  return s

def cs_interpolate(eos, max_idx, p1, matrix_rho, matrix_cs, ds):
  """ """
  segment_mat = []
  segment_mat.append([eos[max_idx, 2], np.sqrt(p1)])

  for i in range(7):
    segment_mat.append([matrix_rho[ds, i], matrix_cs[ds, i]])

  segment_mat = np.array(segment_mat)
  cs_interpolation = interp1d(segment_mat[:,0], segment_mat[:,1], kind='linear', fill_value='extrapolate')

  return cs_interpolation

def EOS_HE(he_eos, eos, max_idx, cs_interpolation, rho_final):
  drho = 1e-5

  M = int((1e-3 - eos[max_idx, 2])/1e-5) + int((1e-2 - 1e-3)/1e-4) + int((rho_final - 1e-2)/1e-3)

  p_last, eps_last, rho_last = eos[max_idx,0], eos[max_idx,1], eos[max_idx,2]

  for i in range(M-1):
    p_next = p_last + cs_interpolation(rho_last)**2 * drho * (eps_last + p_last)/rho_last
    eps_next = eps_last + drho * (eps_last + p_last) / rho_last
    rho_next = rho_last + drho

    if rho_next > rho_final:
      break

    #append row
    he_eos = np.vstack([he_eos, [p_next, eps_next, rho_next]])

    p_last, eps_last, rho_last = p_next, eps_next, rho_next

    if rho_next < 1e-3:
      drho = 1e-5
    elif rho_next < 1e-2:
      drho = 1e-4
    elif rho_next < 1e-1:
      drho = 1e-3
    else:
      drho = 1e-2

  return he_eos

def mergeEOS(eos_mat, he_eos, eos, max_idx):
  eos_mat = np.vstack([eos_mat, eos[:max_idx+1,:]])#vstack is better than the loops
  eos_mat = np.vstack([eos_mat,he_eos])
  return eos_mat


# we combine this in the following function. here we build the equation of state matrix before the lambda transition

# In[9]:


def build(eos_mat, eos, matrix_rho, matrix_cs, ds, rho_treshold, rho_final):
  i = find_pos(eos, rho_treshold, 2)
  p1 = deriv(eos, i)
  cs_interpolation = cs_interpolate(eos, i, p1, matrix_rho, matrix_cs, ds)

  he_eos = np.empty((0,3))
  he_eos = EOS_HE(he_eos, eos, i, cs_interpolation, rho_final)

  eos_matrix = np.empty((0,3))
  eos_matrix = mergeEOS(eos_matrix, he_eos, eos, i)

  return eos_matrix


# In[10]:


Lambda=0
def eos_interpolate(x):
  if x < pc:
    return eps_fluid(x)
  else:
    return eps_fluid(x+Lambda)+Lambda

def eos_prime_interpolate(x):
  if x<pc:
    return eps_prime(x)
  else:
    return eps_prime(x+Lambda)


# In[11]:


def tov_equations(r,u):
  #u=[f,h,P,H,beta]
  f, h, P, H, beta = u

  eps = eos_interpolate(P)
  deps = eos_prime_interpolate(P)

  du1 = (1 - f - 8*np.pi * (r ** 2) * eps) / r
  du2 = -(h * (-1 + f - 8 * np.pi * (r ** 2) * P)) / (r * f)
  du3 = ((-1 + f - 8 * np.pi * (r**2)*P) * (P+eps)) / (2*r*f)
  du4 = beta
  du5 = (H * (- (f**3)
              + (1 + 8 * np.pi * (r**2) * P)**3
              - f * (1 + 8 * np.pi * (r**2) * P) * (-3+60*np.pi * (r**2)*P + 20 * np.pi * (r**2) * eps)
              + (f**2) * (-3
                          + 60 * np.pi * (r**2) * P
                          + 8 * np.pi * (r**3) * deps * (-1 + f - 8 * np.pi * (r**2)*P) * (P+eps)/(2 * r * f)
                          + 20 * np.pi * (r**2) * eps
          )) + r * f * (-1 + f - 8 * np.pi * (r**2) * P) * (1 + f + 4 * np.pi * (r**2) * P - 4 * np.pi * (r**2) * eps) * beta
  ) / ((r**2) * (f**2) * (1 - f + 8 * np.pi * (r**2) * P))

  return [du1, du2, du3, du4, du5]


# In[12]:


def integrator(P0):
  def stop_surface(r,u):
    return u[2] / P0 - 1e-12

  stop_surface.terminal = True
  stop_surface.direction = -1

  u0 = [f0, h0, P0, H0, beta0]

  #using runge kutta method, may change
  solve = solve_ivp(tov_equations, rspan, u0, method="RK45", max_step = 0.05,rtol = 1e-05, atol=1e-7, events=stop_surface)#max_step=5e04
  return solve


# In[13]:


def cycle_tov(data_matrix, P0, Pf):
  if Pf > 1e-2:
    N = int(np.floor( (1e-4 - P0) / (2.5e-6) + (1e-3 - 1e-4)/(2.5e-5) + (1e-2 - 1e-3)/(2.5e-4)))
  elif Pf > 1e-3:
    N = int(np.floor( (1e-4 - P0) / (2.5e-6) + (1e-3 - 1e-4)/(2.5e-5) + (Pf - 1e-3)/(2.5e-4)))
  elif Pf> 1e-4:
    N = int(np.floor((1e-4 - P0)/(2.5e-6) + (Pf - 1e-4)/(2.5e-5)))
  elif Pf > 1e-5:
    N = int(np.floor((Pf - P0)/(2.5e-6)))
  else:
    N = 0

  for i in range(N):
    #print("P0 =", P0)#remove this when code works properly
    solution = integrator(P0)
    Radius = solution.t[-1]#radius
    M = Radius / 2*(1-solution.y[0,-1])#mass
    y = Radius * solution.y[4,-1] / solution.y[3,-1]
    C = M / (Radius / Lu * 1e-5) #compactness

    #love number
    k2 = (8 * ((1 - 2 * C)**2) * (C**5) * (2 + 2 * C * (-1 + y) - y)
          ) / ( 5 * ( 2 * C * (6
                               + (C**2) * (26 - 22 * y)
                               - 3 * y
                               + 4 * (C**4) * (1 + y)
                               + 3 * C * (-8 + 5 * y)
                               + (C**3) * (-4 + 6 * y)
                               ) + 3 * ((1 - 2 * C)**2) * (2 + 2 * C * (-1 + y) - y) * np.log(1 - 2 * C)
                               ))
    lamb = (2/3) * k2 * ((Radius / Lu * 1e-5)**5) * (KK**5) / (M**5)
    row_next = np.array([[P0 / rhou, M, Radius / Lu * 1e-5, lamb]])
    data_matrix = np.vstack([data_matrix, row_next])

    if P0 < 1e-4:
      P0 += 2.5e-6
    elif P0 < 1e-3:
      P0 += 2.5e-5
    elif P0 < 1e-2:
      P0 += 2.5e-4
    else:
      P0 += 2.5e-3

  return data_matrix


# In[14]:


eps_fluid = np.array([])
def eps_prime(x):
  return np.array([])


# In[15]:


def process_one_j(j, eos_name, eos_base, rho_matrix, cs_matrix, out_dir):
    global Lambda, eps_fluid, eps_prime

    z = j
    print(f"starting j = {j+1}")

    local_total = 0
    local_accepted = 0
    saved = []

    eos_matrix = np.empty((0, 3))
    eos_matrix = build(eos_matrix, eos_base, rho_matrix, cs_matrix, z, rho_t, rho_fin)

    p = eos_matrix[:, 0]
    eps = eos_matrix[:, 1]

    eps_fluid_base = interp1d(p, eps, kind='linear', fill_value="extrapolate")
    deps_dp = np.gradient(eps, p)
    eps_prime_base = interp1d(p, deps_dp, kind='linear', fill_value='extrapolate')

    eos_end_p = eos_matrix[-1, 0]

    for i in range(len(Lamb_arr)):
        Lambda = Lamb_arr[i]
        local_total += 1

        eps_fluid = eps_fluid_base
        eps_prime = eps_prime_base

        P0 = 2.5e-5
        if Lambda > 0 and eos_end_p > pc:
            Pf = eos_end_p - Lambda
        else:
            Pf = eos_end_p

        data_matrix = np.empty((0, 4))
        data_matrix = cycle_tov(data_matrix, P0, Pf)

        if data_matrix.size == 0:
            continue

        M_max = np.max(data_matrix[:, 1])

        if Lambda == 0:
            temp = 0
        else:
            temp = int(np.floor((abs(Lambda / (Kp * Pu)) ** 0.25) / np.sign(Lambda)))

        if 2.18 < M_max < 2.52:
            local_accepted += 1
            out_path = f"{out_dir}/TOV_{eos_name}_{temp}_{z+1}.csv"
            np.savetxt(out_path, data_matrix)
            saved.append(out_path)

    print(f"done j = {j+1}")
    return {"total": local_total, "accepted": local_accepted, "saved": saved}


# In[16]:


def generate_tovs(eos_name, eos_base, rho_matrix, cs_matrix, out_dir, n_jobs=6):
    os.makedirs(out_dir, exist_ok=True)

    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(process_one_j)(j, eos_name, eos_base, rho_matrix, cs_matrix, out_dir)
        for j in range(rho_matrix.shape[0])
    )

    total = sum(r["total"] for r in results)
    accepted = sum(r["accepted"] for r in results)

    print("eos tested:", total)
    print("eos accepted:", accepted)
    print("acceptance rate:", accepted / total if total > 0 else np.nan)


# Below is for (M,R) data only

# In[21]:


def data_generator(R_rand, M_rand, data_matrix, idx_max, sigma_M, sigma_R, TOT):
    M_arr = data_matrix[:, 1]
    R_arr = data_matrix[:, 2]

    M_unique, idx_unique = np.unique(M_arr[:idx_max+1], return_index=True)
    R_unique = R_arr[:idx_max+1][idx_unique]

    RM_curve = interp1d(M_unique, R_unique, kind='linear', fill_value='extrapolate')
    M_rand = uniform.rvs(loc=M_unique[0], scale=M_unique[-1] - M_unique[0], size=TOT)
    R_rand = RM_curve(M_rand)

    M_rand = norm.rvs(loc=M_rand, scale=sigma_M) / M_norm
    R_rand = norm.rvs(loc=R_rand, scale=sigma_R) / R_norm

    return R_rand, M_rand


# In[22]:


def generate_mr_dataset(source_dir, dataset_path, rho_matrix, cs_matrix, ns=300, TOT=30):
    sigma_M = 0.1
    sigma_R = 0.5
    M_norm = 3
    R_norm = 20

    if os.path.exists(dataset_path):
        os.remove(dataset_path)

    listfile = sorted(os.listdir(source_dir))

    for file_idx, filename in enumerate(listfile, start=1):
        print(f"{file_idx}/{len(listfile)} : {filename}")

        parts = filename.split(".")[0].split("_")
        eos_name = parts[1]
        Lambda_temp = int(parts[2])
        z = int(parts[3])

        TOV_matrix = np.loadtxt(os.path.join(source_dir, filename))

        rows_to_write = []

        for _ in range(ns):
            R_rand = np.array([])
            M_rand = np.array([])

            idx_max = np.argmax(TOV_matrix[:, 1])
            R_rand, M_rand = data_generator(
                R_rand, M_rand, TOV_matrix, idx_max, sigma_M, sigma_R, TOT
            )

            row = np.concatenate([
                np.array([eos_name, Lambda_temp], dtype=object),
                rho_matrix[z - 1, :],
                cs_matrix[z - 1, :],
                M_rand,
                R_rand
            ])

            rows_to_write.append(row)

        rows_to_write = np.array(rows_to_write, dtype=object)

        with open(dataset_path, "ab") as f:
            np.savetxt(f, rows_to_write, fmt="%s")


# In[24]:


def data_generator_k2(R_rand, M_rand, k2_rand, data_matrix, idx_max, TOT, sigma_M, sigma_R, sigma_k2):
    # columns in data_matrix: [P0, M, R, lambda]
    M_arr = data_matrix[:, 1]
    R_arr = data_matrix[:, 2]
    lamb_arr = data_matrix[:, 3]

    # compute k2 from lambda
    k2_arr = (3.0 / 2.0) * lamb_arr * (M_arr ** 5) / (R_arr ** 5) / (KK ** 5)

    # keep stable branch up to maximum mass, deduplicate mass values
    M_unique, idx_unique = np.unique(M_arr[:idx_max+1], return_index=True)
    R_unique = R_arr[:idx_max+1][idx_unique]
    k2_unique = k2_arr[:idx_max+1][idx_unique]

    RM_curve = interp1d(M_unique, R_unique, kind="linear", fill_value="extrapolate")
    k2M_curve = interp1d(M_unique, k2_unique, kind="linear", fill_value="extrapolate")

    # sample masses uniformly on stable branch
    M_rand = uniform.rvs(loc=M_unique[0], scale=M_unique[-1] - M_unique[0], size=TOT)
    R_rand = RM_curve(M_rand)
    k2_rand = k2M_curve(M_rand)

    # inject Gaussian noise
    M_rand = norm.rvs(loc=M_rand, scale=sigma_M) / M_norm
    R_rand = norm.rvs(loc=R_rand, scale=sigma_R) / R_norm
    k2_rand = norm.rvs(loc=k2_rand, scale=sigma_k2)   # normalization done later, like paper

    return R_rand, M_rand, k2_rand


# In[25]:


def generate_mrk2_dataset(source_dir, dataset_k2_path, rho_matrix, cs_matrix, ns=100, TOT=30):
    sigma_k2 = 0.05
    sigma_M = 0.1
    sigma_R = 0.5
    M_norm = 3
    R_norm = 20

    if os.path.exists(dataset_k2_path):
        os.remove(dataset_k2_path)

    listfile = sorted(os.listdir(source_dir))

    for file_idx, filename in enumerate(listfile, start=1):
        print(f"{file_idx}/{len(listfile)} : {filename}")

        parts = filename.split(".")[0].split("_")
        eos_name = parts[1]
        Lambda_temp = int(parts[2])
        z = int(parts[3])

        TOV_matrix = np.loadtxt(os.path.join(source_dir, filename))

        rows_to_write = []

        idx_max = np.argmax(TOV_matrix[:, 1])

        for _ in range(ns):
            R_rand = np.array([])
            M_rand = np.array([])
            k2_rand = np.array([])

            R_rand, M_rand, k2_rand = data_generator_k2(
                R_rand, M_rand, k2_rand,
                TOV_matrix, idx_max, TOT,
                sigma_M, sigma_R, sigma_k2
            )

            row = np.concatenate([
                np.array([eos_name, Lambda_temp], dtype=object),
                rho_matrix[z - 1, :],
                cs_matrix[z - 1, :],
                M_rand,
                R_rand,
                k2_rand
            ])

            rows_to_write.append(row)

        rows_to_write = np.array(rows_to_write, dtype=object)

        with open(dataset_k2_path, "ab") as f:
            np.savetxt(f, rows_to_write, fmt="%s")

