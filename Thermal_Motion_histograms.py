import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
from pylab import loadtxt
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

# ============== setting up files ===============
filenames = ["1-1_data.txt", "1-2_data.txt", "2-1_data.txt", "2-2_data.txt", "3-1_data.txt", "4-1_data.txt", "4-2_data.txt", "5-1_data.txt", "5-2_data.txt", "6-1_data.txt", "6-2_data.txt", "7-1_data.txt", "8-1_data.txt", "8-2_data.txt", "9-1_data.txt", "9-2_data.txt", "10-1_data.txt", "10-2_data.txt", "11-1_data.txt", "11-2_data.txt"]


# ============= defining functions ==============
def convert(values_list):
  '''Used for converting pixels to micrometers'''
  new_list = []
  for i in range(len(values_list)):
    new_list.append(values_list[i]* 0.1155)
  return new_list

def findTotalDistance(x_values, y_values):
  '''Finding total distance from initial point to point i, and returns it as a list. If the x-value or y-value is zero, distance is 0 by default'''
  dist_array = [];
  for i in range(1, len(x_values)):
    x_diff = x_values[i]-x_values[0]
    y_diff = y_values[i]-y_values[0]
    dist = (x_diff**2 + y_diff**2) # ||r - r_0||^2
    if (x_values[i]==0) | (y_values[i]==0):
      dist = 0
    dist_array.append(dist)
  return dist_array

def findDiff(values):
  '''Finds the total difference between an initial value and value i'''
  dist_array = []
  for i in range(1, len(values)):
    if values[i] != 0:
      diff = values[i] - values[0]
      dist_array.append(diff)
    else:
      dist_array.append(0)
  return dist_array
  
def findDistance(values):
  '''Finds the difference between consecutive steps and returns the results as an array'''
  dist_array = []
  for i in range(1, len(values)):
    if values[i] != 0:
      diff = values[i] - values[i-1]
      dist_array.append(diff)
  return dist_array


def averageDistance(dist_list):
  '''Taking in a list of lists with j signifying each trial and i signifying each time step in the trial, finds the average distance for each time step and returns it as an array'''
  dist_avg_list = []
  num = len(dist_list)
  for i in range(119):
    ind_avg = 0
    num_eff = 0
    for j in range(num):
      if dist_list[j][i] !=0:
        ind_avg += dist_list[j][i]
        num_eff += 1
    ind_avg /= num_eff
    dist_avg_list.append(ind_avg)

  return dist_avg_list

def r2Uncertainty(x_data, y_data):
  '''Calculates uncertainty for each <r^2> data point'''
  # x_data is a list of x diff across all trials
  # y_data is same

  N = 20
  res = 0

  for i in range(20):
    if (x_data[i]==0)|(y_data[i]==0):
      N-=1
    else:
      res += (2*x_data[i]*0.14)**2+(2*y_data[i]*0.14)**2

  res /= N
  
  return res**0.5

def maxLikelihood(r_data):
  '''Takes in stepwise r data and returns 2Dt'''
  sum = 0
  for i in range(len(r_data)):
    sum += r_data[i]**2
  res = sum/(2*len(r_data))
  return res

def rUncertainty(r_data):
  '''Takes r in 0.5s interval data and returns uncertainty for 2Dt'''
  sum = 0
  for i in r_data:
    sum += (2*i*0.14)**2
  sum /= 2*len(r_data)
  return sum**0.5


def chisq(obs_a, exp_a):
  sum = 0
  for i in range(len(obs_a)):
    sum += ((obs_a[i]-exp_a[i])**2)/exp_a[i]
  return sum

# ============== processing data ===================
dist_data_total = []
dist_data_x = []
dist_data_y = []
dist_data_r = []
diff_data_x = []
diff_data_y = []

for j in range(len(filenames)):
  filename = filenames[j]
  data=loadtxt(filename, usecols=(0,1), skiprows=2, unpack=True)
  x_data = convert(data[0]) # x data in micrometers
  y_data = convert(data[1]) # y data in micrometers
  dist_data = []
  dist_data_x.extend(findDistance(x_data)) # cumulative list of all the different steps in x
  dist_data_y.extend(findDistance(y_data)) # same for y
  dist_data = findTotalDistance(x_data, y_data) # calculates the total distance between the start and a point for every time step
  dist_data_total.append(dist_data) # adding the list to a list of lists
  #print(dist_data_r2[len(dist_data_r2)-1])

  diff_data_x.append(findDiff(x_data)) # adding list to list of lists
  diff_data_y.append(findDiff(y_data))


uncertainties = []

'''Calculating uncertainties for every data point'''
for i in range(len(diff_data_x[0])):
  diff_datax2 = []
  diff_datay2 = []
  for j in range(len(diff_data_x)):
    diff_datax2.append(diff_data_x[j][i])
    diff_datay2.append(diff_data_y[j][i])
  uncertainties.append(r2Uncertainty(diff_datax2, diff_datay2))

'''Finding distance travelled between consecutive points'''
for i in range(len(dist_data_x)):
  if dist_data_x[i] != 0:
    dist_data_r.append((dist_data_x[i]**2+dist_data_y[i]**2)**0.5)

for i in range(len(dist_data_r)):
  dist_data_r[i] = np.abs(dist_data_r[i])

avg_dist = averageDistance(dist_data_total)
time = np.linspace(0, 60, 119)


# ============ curve-fitting ==============
xerror=0.03
yerror=uncertainties
# finished importing data, naming it sensibly

def my_func(x,m):
    return m*x
# this is the function we want to fit. the first variable must be the
# x-data (time), the rest are the unknown constants we want to determine

init_guess=(20)
# your initial guess of (a,tau,T,phi)

popt, pcov = optimize.curve_fit(my_func, time, avg_dist, p0=init_guess, maxfev=10000)
# we have the best fit values in popt[], while pcov[] tells us the uncertainties

m=popt[0]
# best fit values are named nicely
u_m=pcov[0,0]**(0.5)
# uncertainties of fit are named nicely

def fitfunction(x):
    return m*x 
#fitfunction(t) gives you your ideal fitted function, i.e. the line of best fit

start=min(time)
stop=max(time)
xs=np.arange(start,stop,(stop-start)/1000) # fit line has 1000 points
curve=fitfunction(xs)
# (xs,curve) is the line of best fit for the data in (xdata,ydata)

print(chisq(avg_dist[1:], fitfunction(time)[1:]))



# =============== maximum likelihood estimate ===========
# PROBLEMS WITH CALCULATIONS HERE!!!
T, n, r = 296.5, 0.001, 0.95*(10**(-6))
gamma = 6*np.pi*n*r

# p(r;t) = r/2Dt *exp(-r^2/4Dt)
'''
r_data=[]
for i in dist_data_r:
  if (i<=3)&(i>=-3):
    r_data.append(i)

r_data=dist_data_r

hist, bin_edges = np.histogram(r_data, bins=50)
hist=hist/sum(hist)

n = len(hist)
r_hist=np.zeros((n),dtype=float) 
for ii in range(n):
    r_hist[ii]=(bin_edges[ii+1]+bin_edges[ii])/2
    
y_hist=hist
       
#Calculating the Gaussian PDF values given Gaussian parameters and random variable X
def gaus(R,D):
    t = 0.5
    return exp(-(R**2)/(4*D*t))*R/(2*D*t)

mean = sum(r_hist*y_hist)/sum(y_hist)                  
sigma = sum(y_hist*(r_hist-mean)**2)/sum(y_hist) 

#Gaussian least-square fitting process
param_optimised,param_covariance_matrix = curve_fit(gaus,r_hist,y_hist,p0=[1e-12],maxfev=5000)

#print fit Gaussian parameters
print("Fit parameters: ")
print("=====================================================")
print("D = ", param_optimised[0], "+-",np.sqrt(param_covariance_matrix[0,0]))
'''



# ============ calculating k ============
print(m)
D_3 = m*(10**(-12))/4
print(D_3)
k_D = D_3*gamma/T
print(k_D)

# ============ max likelihood ==============
D_4 = maxLikelihood(dist_data_r)*10**-12
print(D_4)
m_D = D_4*gamma/T
print(m_D)

# ================ graphing ==============
#fig, (ax1,ax3,ax4) = plt.subplots(1,3)

#ax1.hist(dist_data_x, 50)
#ax1.set_title("x-direction movement histogram") 
#ax1.set_xlabel("Distance (μm)")
#ax1.set_ylabel("# of instances")
#ax1.set(xlim=(-10, 10), ylim=(0, 40))
'''
#STEP 4: PLOTTING THE GAUSSIAN CURVE -----------------------------------------
#fig, (ax1) = plt.subplots(1,1)
r_hist_2=np.linspace(np.min(r_hist),np.max(r_hist),500)
#ax1.plot(r_hist_2,gaus(r_hist_2,*param_optimised),'r.:',label='Gaussian fit')
#ax1.legend()

#Normalise the histogram values
weights = np.ones_like(r_data) / len(r_data)
ax1.hist(r_data, bin_edges, weights=weights)

#setting the label,title and grid of the plot
ax1.set_xlabel("Distance travelled in 0.5s (μm)")
ax1.set_ylabel("Probability")
ax1.set_title("Histogram of distance travelled in a 0.5s interval")
'''

fig, (ax1,ax2) = plt.subplots(1,2)

r=dist_data_r
    
# Histogram
hist_values, bin_edges, patches = ax1.hist(r, bins=50, density=True, label='Counts')

bin_centers = (bin_edges[1:] + bin_edges[:-1])/2

x = bin_centers[:]  # not necessary, and I'm not sure why the OP did this, but I'm doing this here because OP does
y = hist_values[:]

def f(R, D):
    t=0.5
    return R/(2*D*t)*np.exp(-1*(R**2)/(4*D*t))

popt, pcov = optimize.curve_fit(f, x, y, p0 = 0.1, maxfev=10000)

D5=popt[0]
# best fit values are named nicely
u_D5=pcov[0,0]**(0.5)
# uncertainties of fit are named nicely

def fitfunction2(R):
    t=0.5
    return R/(2*D5*t)*np.exp(-1*(R**2)/(4*D5*t))
#fitfunction(t) gives you your ideal fitted function, i.e. the line of best fit

start2=min(x)
stop2=max(x)
xs2=np.arange(start2,stop2,(stop2-start2)/1000) # fit line has 1000 points
curve2=fitfunction2(xs2)
# (xs,curve) is the line of best fit for the data in (xdata,ydata)


print(D5)
print(u_D5)

ax1.plot(xs2, curve2, label='PDF', color='orange')

ax1.set_title('Histogram of Distances Travelled in 0.5s')
ax1.set_ylabel('Relative Requency')
ax1.set_xlabel('Distance (μm)') # Motion seems to be in micron range, but calculation and plot has been done in meters

D_5 = D5*10**-12

f_D = D_5*gamma/T
print(f_D)

residual=y-fitfunction2(x)
# find the residuals
zeroliney=[0,0]
zerolinex=[start2,stop2]
# create the line y=0

ax2.errorbar(x,residual,yerr=0,xerr=0,fmt=".")
ax2.plot(zerolinex,zeroliney)

# plotnthe y=0 line on top

ax2.set_xlabel("Distance (μm)")
ax2.set_ylabel("Relative Frequency")
ax2.set_title("Residuals of the fit")

obs_a = y
exp_a = fitfunction2(x)

#ax2.hist(dist_data_y, 50)
#ax2.set_title("y-direction movement histogram")
#ax2.set_xlabel("Distance (μm)")
#ax2.set_ylabel("# of instances")
'''
ax3.errorbar(time,avg_dist,yerr=yerror,xerr=xerror,fmt=".")
# plot the data, fmt makes it data points not a line
ax3.plot(xs,curve)
# plot the best fit curve on top of the data points as a line

ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Mean squared distance (μm^2)")
ax3.set_title("Mean squared distance of particles vs. Time")
# HERE is where you change how your graph is labelled


print("M:", m, "+/-", u_m)
# prints the various values with uncertainties

residual=avg_dist-fitfunction(time)
# find the residuals
zeroliney=[0,0]
zerolinex=[start,stop]
# create the line y=0

ax4.errorbar(time,residual,yerr=yerror,xerr=xerror,fmt=".")
# plot the residuals with error bars
ax4.plot(zerolinex,zeroliney)
# plotnthe y=0 line on top

ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Mean squared distance (μm^2)")
ax4.set_title("Residuals of the fit")
# HERE is where you change how your graph is labelled
'''
plt.show()
