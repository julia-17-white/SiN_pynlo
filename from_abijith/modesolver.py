import numpy as np
import EMpy
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light, epsilon_0

def sinIndex(lda):
    n=np.sqrt((1+3.0249/(1-(0.1353406/lda)**2)+40314/(1-(1239.842/lda)**2)))
    return n


def sio2Index(lda):
    n=np.sqrt((1+0.6961663/(1-(0.0684043/lda)**2) \
               + 0.4079426/(1-(0.1162414/lda)**2) \
               + 0.8974794/(1-(9.896161/lda)**2)))
    return n


def nbk7(x):
    n=(1+1.03961212/(1-0.00600069867/x**2) \
       +0.231792344/(1-0.0200179144/x**2) \
       +1.01046945/(1-103.560653/x**2))**.5
    return n


def n2(x,y):
    xx, yy = np.meshgrid(x,y)
    
    return np.where((np.abs(xx.T) <= width/2.0) *
                      (np.abs(yy.T) <= height/2.0),
                      3.6e-7,
                      2e-8)


def modeparams(Ex,solver_x,solver_y,wl,neff):
    
    c = speed_of_light
    e0 = epsilon_0
    
    Ex_i = EMpy.utils.interp2(x,y,EMpy.utils.centered1d(solver_x),EMpy.utils.centered1d(solver_y),Ex)
    I = 0.5*e0*epsfunc(x,y,wl)*c*np.abs(Ex_i)**2

    n2eff_numr=np.trapezoid(np.trapezoid(np.sqrt(epsfunc(x,y,wl))*n2(x,y)*I**2,x),y)
    n2eff_denr=neff*np.trapezoid(np.trapezoid(I**2,x),y)

    n2eff=n2eff_numr/n2eff_denr
    Aeff = np.trapezoid(np.trapezoid(I,x),y)**2/(np.trapezoid(np.trapezoid(I**2,x),y))
    w = 2*np.pi*c/wl*1e6
    gamma = n2eff*w/(c*Aeff)
    
    return Aeff,gamma


c = speed_of_light
ldas = np.linspace(0.4,2.4,50)

#ldas = np.array([1.55])

#ldas = np.linspace(1., 2., 20)
#ldas = np.array([])
x = np.linspace(-4, 4, 400)
y = np.linspace(-4, 4, 400)

neigs = 1
tol = 1e-4
boundary = '000S'

# widths = np.linspace(0.8,5.,43)
widths = np.array([1.0])

import time

height = 0.8
indx = 0 # indexing the print statements to give an idea of the run time


for j in range(len(widths)):
    neffs = 0*ldas
    Aeffs = 0*ldas
    gammas = 0*ldas
    width = widths[j]
    
    def epsfunc(x_, y_, wl):
        xx, yy = np.meshgrid(x_, y_)
        return np.where((np.abs(xx.T) <= width/2.0) *
                      (np.abs(yy.T) <= height/2.0),
                      sinIndex(wl)**2,
                      1.0) # sio2Index(wl)**2
    
    
    for k in range(len(ldas)):
        indx += 1
        print(f'For loop {indx} our of {len(widths)*len(ldas)}:')
        wl = ldas[k]
        efunc = lambda X, Y: epsfunc(X, Y, wl)
        start = time.time()
        solver = EMpy.modesolvers.FD.SVFDModeSolver(wl, x, y, efunc, boundary,
                                                    method='Ex').solve(neigs, tol)
        print(f'run time: {time.time()-start}')
        Aeffs[k],gammas[k] = modeparams(solver.Ex[0],solver.x,solver.y,wl,solver.neff[0])
        
        neffs[k] = solver.neff[0]
        print('(wavelength, n_eff, gamma, a_eff):')
        print(wl, neffs[k], gammas[k], Aeffs[k])
        print('')
    
    np.save('from_abijith/jw_modes/JW_SiN_AirClad_800nmThickness_' + str(int(width*1000)) + 'nmWidth_gamma_aeff.npy',np.column_stack([ldas,neffs,gammas,Aeffs]))


dlda = np.diff(ldas)[0]
D = -1e12*(ldas[:-2]/c)*np.diff(np.diff(neffs[:])/dlda)/dlda


# fig, ax = plt.subplots(tight_layout=True)

# Ey_i = EMpy.utils.interp2(x,y,EMpy.utils.centered1d(solver.x),EMpy.utils.centered1d(solver.y),solver.Ey[0])

# cf = ax.contourf(x,y,np.abs(Ey_i.T), 50)
# ax.set_xlabel(r'X [$\mu$m]')
# ax.set_ylabel(r'Y [$\mu$m]')
# cb = plt.colorbar(cf)
# th = np.linspace(0, 2*np.pi, 100)

# ax.plot(4*np.cos(th),3*np.sin(th),'-r',linewidth=2)


### Dispersion Curve
# plt.figure()
# from scipy.interpolate import UnivariateSpline, interp1d, \
#     InterpolatedUnivariateSpline
# from scipy.constants import speed_of_light
# wlMin = 0.4
# wlMax = 2
# #lda_, neff_0, neff_1 = np.loadtxt(fileName,delimiter=',',unpack=True)

# lda = np.linspace(wlMin, wlMax, 5000)
# s = InterpolatedUnivariateSpline(ldas, neffs)

# n_0 = s(lda)

# #plt.plot(lda, n_0, lda_, neff_0)

# dn_dlda = np.gradient(n_0)/np.gradient(lda)
# c = speed_of_light
# vg = c/(n_0 - lda*dn_dlda)

# D = np.gradient(1/vg)/np.gradient(lda)

# plt.plot(lda, D*1e12)
