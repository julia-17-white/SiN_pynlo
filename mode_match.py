# %% Imports ==================================================================
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import os, sys
from scipy.optimize import curve_fit
from scipy import constants
import multiprocessing
import traceback
import colorcet
import EMpy
import EMpy.modesolvers.FD

pi = constants.pi
c = constants.c

# %% RI Data ==================================================================

#--- Sellmeier Equation
def sellmeier(wvl, *coefs):
    n2 = 1
    n = len(coefs)//2
    coefs = np.reshape(coefs, (n, 2))
    for coef in coefs:
        n2 += coef[1]/(1-(coef[0]/wvl)**2)
    return abs(n2)**0.5

def fit_sellmeier(wvl_data, ri_data, n=3):
    coefs = []
    for coef in range(n):
        x_res = 10*np.max(wvl_data) if coef % 2 else 0.1*np.min(wvl_data)
        test_coef = [x_res, 0.5]
        bounds = ([0.]*(coef+1)*2, [np.inf]*(coef+1)*2)
        #print(coefs+test_coef)
        popt, pcov = curve_fit(sellmeier, wvl_data, ri_data, p0=coefs+test_coef, bounds=bounds)
        coefs = popt.tolist()
    return coefs

#--- Literature Values
def vacuum(wvl):
    return 1. + 0*wvl

def SiO2(wvl):
    """Refractive index of SiO2, laser wavelength in meters. Malitson"""
    x = 1e6*wvl
    return (1+0.6961663/(1-(0.0684043/x)**2)+0.4079426/(1-(0.1162414/x)**2)+0.8974794/(1-(9.896161/x)**2))**.5

def SiN(wvl):
    """
    Refractive index of SiN, laser wavelength in meters
    K. Luke, et. al. Broadband mid-infrared frequency comb generation in a Si3N4 microresonator, Opt. Lett. 40, 4823-4826 (2015)
    """
    x = wvl*1e6 # convert wavelength units to microns
    # eps = 0 # LPCVD
    eps = 0.05 # PECVD
    return np.sqrt(1.+(1-eps)*(3.0249/(1.-(0.1353406/x)**2.)+40314./(1.-(1239.842/x)**2.)))


#--- LIGENTEC RI Data
# ligentec_SiN = np.genfromtxt(r"/Users/pooja/Documents/Research/Work/Python Scripts/reference_SiN_lossless.txt", usecols=(0,1), unpack=True)[:, ::-1]
# ligentec_SiN_sel = fit_sellmeier(1e-3*ligentec_SiN[0], ligentec_SiN[1], n=3)
# def SiN_LIGENTEC(wvl):
#     """ Refractive index of SiN as measured by LIGENTEC, laser wavelength in meters.
#     """
#     x = wvl*1e6
#     return sellmeier(x, *ligentec_SiN_sel)

try:
    ligentec_SiN = np.genfromtxt("reference_SiN_lossless.txt", usecols=(0,1), unpack=True)[:, ::-1]
    ligentec_SiN_sel = fit_sellmeier(1e-3*ligentec_SiN[0], ligentec_SiN[1], n=3)

    def SiN_LIGENTEC(wvl):
        x = wvl * 1e6
        return sellmeier(x, *ligentec_SiN_sel)

    nfunc = SiN_LIGENTEC
    print("Using LIGENTEC SiN refractive index")

except FileNotFoundError:
    nfunc = SiN
    print("LIGENTEC data missing — using Luke SiN model")


'''
plt.figure("SiN RI")
plt.clf()
x_samp = np.linspace(ligentec_SiN[0].min()/2, ligentec_SiN[0].max()*2, 10000)

plt.plot(x_samp, SiN(1e-9*x_samp), label="Luke")

plt.plot(ligentec_SiN[0], ligentec_SiN[1], '.', label="LIGENTEC")

sel_fit = fit_sellmeier(1e-3*ligentec_SiN[0], ligentec_SiN[1], n=3)
print(sel_fit)
plt.plot(x_samp, sellmeier(1e-3*x_samp, *sel_fit), label="Sellmeier Fit")

plt.xlim(ligentec_SiN[0].min(), ligentec_SiN[0].max())
plt.ylim(ligentec_SiN[1].min(), ligentec_SiN[1].max())
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Refractive Index")
plt.title("SiN")
plt.grid(True)
plt.tight_layout()
'''

# ligentec_SiO2 = np.genfromtxt(r"/Users/pooja/Documents/Research/Work/Python Scripts/reference_SiO2_lossless.txt", usecols=(0,1), unpack=True)[:, ::-1]
# ligentec_SiO2_sel = fit_sellmeier(1e-3*ligentec_SiO2[0], ligentec_SiO2[1], n=1)
# def SiO2_LIGENTEC(wvl):
#     """ Refractive index of SiO2 as measured by LIGENTEC, laser wavelength in meters.
#     """
#     x = wvl*1e6
#     return sellmeier(x, *ligentec_SiO2_sel)

try:
    ligentec_SiN = np.genfromtxt("reference_SiN_lossless.txt", usecols=(0,1), unpack=True)[:, ::-1]
    ligentec_SiN_sel = fit_sellmeier(1e-3*ligentec_SiN[0], ligentec_SiN[1], n=3)

    def SiN_LIGENTEC(wvl):
        x = wvl * 1e6
        return sellmeier(x, *ligentec_SiN_sel)

    nfunc = SiN_LIGENTEC
    print("Using LIGENTEC SiN refractive index")

except FileNotFoundError:
    nfunc = SiN
    print("LIGENTEC data missing — using Luke SiN model")


'''
plt.figure("SiO2 RI")
plt.clf()
x_samp = np.linspace(ligentec_SiN[0].min()/2, ligentec_SiN[0].max()*2, 10000)

plt.plot(x_samp, SiO2(1e-9*x_samp), label="Malitson")

plt.plot(ligentec_SiO2[0], ligentec_SiO2[1], '.', label="LIGENTEC")

sel_fit = fit_sellmeier(1e-3*ligentec_SiO2[0], ligentec_SiO2[1], n=1)
print(sel_fit)
plt.plot(x_samp, sellmeier(1e-3*x_samp, *sel_fit), label="Sellmeier Fit")

plt.xlim(ligentec_SiO2[0].min(), ligentec_SiO2[0].max())
plt.ylim(ligentec_SiO2[1].min(), ligentec_SiO2[1].max())
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Refractive Index")
plt.title("SiO2")
plt.grid(True)
plt.tight_layout()
'''

#--- Experimental RI Data
def AlN_refractive_index(wvl):
    """
    this returns the refractive index for AlN. Two arrays are returned, one for the ordinary
    and one for the extraordinary.

    Parameters
    ----------
    x : float
        the wavelength in meters

    Returns
    -------
    no : float
        The ordinary refractive index
    ne : float
        The extraordinary refractive index
    """
    x = wvl*1e6

    def RI(x, no, ba1, ba2, ca1, ca2):
        """This is the equation that Hojoong Jung used to fit his refractive index data."""

        return (no + ba1*x**2/(x**2-ca1**2) + ba2*x**2/(x**2-ca2**2) )**0.5

    no = RI(x, no=1, ba1=2.96184, ba2=67.69201, ca1=0.13671, ca2=56.49456)
    ne = RI(x, no=1, ba1=3.13801, ba2=11.54373, ca1=0.13460, ca2=23.22600)

    return no, ne

# def nfunc(laser):
#     ne, no = AlN_refractive_index(laser)
#     return np.array((no, 0, 0, ne, no))
#
# def n1func(laser):
#     n = SiO2(laser)
#     return np.array((n, 0, 0, n, n))

#%% Mode Solver ===============================================================

def eps_rectangle(
    xc, yc, height=0.6e-6, width=1.2e-6, n0=1.0, n1=1.45, n2=2.0):
    """
    Return a matrix of relative permittivity values for a rectangular
    waveguide.

    Parameters
    ----------
    xc, yc : array of floats
        x and y values representing the midpoints of each gridbox
    height : float
        the height (y dimension) of the rectangular waveguide
    width : float
        the width (x dimension) of the rectangualar waveguide.
    n0 : float
        the refractive index of the cladding above the waveguide
    n1 : float
        refractive index of the substrate below the waveguide
    n2 : float
        refractive index of the waveguide

    Notes
    -----
    The midpoint of the rectangular waveguide is at the origin of the
    coordinate system. The substrate is defined as the material below
    the waveguide (y<-`height`), and the cladding is the remainder.

    Returns
    -------
    Z:
        2d-matrix of epsilon values. Must be same shape as the transpose of the
        xx, yy from np.meshgrid
    """
    xx_c, yy_c = np.meshgrid(xc, yc)
    xx_c = xx_c.T
    yy_c = yy_c.T

    #--- Isotropic or Anisotropic
    if (isinstance(n0, np.ndarray) or isinstance(n1, np.ndarray) or isinstance(n2, np.ndarray)):
        n_num = np.max((n0.shape[0], n1.shape[0], n2.shape[0]))
    else:
        n_num = 1

    if n_num ==1:
        Z = np.zeros(xx_c.shape)
    elif n_num==5:
        Z = np.zeros(np.append(xx_c.shape, 5))
    else:
        raise ValueError('Refractive indices must have length of 1 or 5.\n Shapes given: %i, %i, %i'%(n0.shape[0], n1.shape[0], n2.shape[0]))

    #--- Cladding
    Z[:] = n0**2

    #--- Substrate
    is_substrate_y = yy_c < -height/2
    Z[is_substrate_y]  = n1**2

    #--- Waveguide
    is_waveguide_x = np.abs(xx_c) < width/2
    is_waveguide_y = np.abs(yy_c) < height/2
    Z[is_waveguide_x & is_waveguide_y] = n2**2

    return Z

def rectangular_waveguide_modes(
    frq, neigs=4, height=600e-9, width=1200e-9, n0=1.0, n1=1.45, n2=2.0,
    grid=[3e-6,3e-6,100e-9,100e-9], boundary='0000', plot_directory=None,
    rel_tol = 0):
    """
    This function finds the modes of a rectangular waveguide using the
    full-vector modesolver implemented in EMPy.

    Parameters
    ----------
    frq : float
        frequency of the light, in Hz.
    neigs : float
        The number of eigenmodes calculated. Eigenmodes are found in order of
        decreasing refractive index.
    height : float
        the height (y dimension) of the rectangular waveguide, in m.
    width : float
        the width (x dimension) of the rectangualar waveguide, in m.
    n0 : float
        the refractive index of the cladding above the waveguide
    n1 : float
        refractive index of the substrate below the waveguide
    n2 : float
        refractive index of the waveguide
    grid : [x, y, dx, dy] or (x_grid, y_grid)
        Information that defines the simulation grid.

        x, y: float
            The x and y distance from the center to the edge of the simulation
        dx, dt: float
            The distance between x and y grid points for the simulation.
            Smaller is more accurate, but takes longer. It's likely good to
            ensure that there are at least ~10 grid points along each dimension
            of the waveguide.
        x_grid, y_grid: array_like
            The x and y coordinates that form the vertices of the grid
    boundary : str
        A string that identifies the type of boundary conditions.
        The following options are available:
           'A' - Hx is antisymmetric, Hy is symmetric.
           'S' - Hx is symmetric and, Hy is antisymmetric.
           '0' - Hx and Hy are zero immediately outside of the boundary.
        The string identifies all four boundary conditions, in the order: north,
        south, east, west. Default is '0000', which does not apply symmetry.
    mode_plot_directory : str
        identifies the directory where the mode plots should be saved. If None,
        then no plots will be saved

    Notes
    -----
    The midpoint of the rectangular waveguide is at the origin of the
    coordinate system. The substrate is defined as the material below
    the waveguide (y<-`height`), and the cladding is the remainder.

    Returns
    -------
    modes : tuple
        the results

    """
    #--- Vacuum Wavelength
    wl = c/frq

    #--- Parse Grid
    if len(grid) == 4:
        x = np.linspace(-grid[0], grid[0], 2*grid[0]/grid[2] + 1)
        y = np.linspace(-grid[1], grid[1], 2*grid[1]/grid[3] + 1)
    elif len(grid) == 2:
        x, y = grid
        x = np.asarray(x)
        y = np.asanyarray(y)
    else:
        raise ValueError("grid is ill-defined")

    # Grid Vertices
    xx, yy = np.meshgrid(x, y)
    xx = xx.T
    yy = yy.T

    # Grid Midpoints
    x_m = (x[1:] + x[:-1]) / 2.
    y_m = (y[1:] + y[:-1]) / 2.
    xx_m, yy_m = np.meshgrid(x_m, y_m)
    xx_m = xx_m.T
    yy_m = yy_m.T

    dx_m = np.diff(x)
    dy_m = np.diff(y)
    dxx, dyy = np.meshgrid(dx_m, dy_m)
    dxx = dxx.T
    dyy = dyy.T
    dxxyy = dxx*dyy

    #--- EMpy Mode Solver
    def eps_function(xc, yc):
        eps = eps_rectangle(
            xc, yc, height=height, width=width, n0=n0, n1=n1, n2=n2)
        return eps

    mode_solver = EMpy.modesolvers.FD.VFDModeSolver(
        wl, x, y, eps_function, boundary)

    # Solve It!
    mode_solver.solve(neigs, tol=rel_tol)
    modes = mode_solver.modes

    #--- Refractive Index
    n_eff = np.array([np.real(n.neff) for n in modes])

    #--- Electric Field
    Ex = np.array([mode.get_field("Ex", x=x_m, y=y_m) for mode in modes])
    Ey = np.array([mode.get_field("Ey", x=x_m, y=y_m) for mode in modes])
    Ez = np.array([mode.get_field("Ez", x=x_m, y=y_m) for mode in modes])

    E2x = Ex.real**2 + Ex.imag**2
    E2y = Ey.real**2 + Ey.imag**2
    E2z = Ez.real**2 + Ez.imag**2
    E2t = E2x + E2y + E2z

    Ex2_sum = np.sum(E2x * dxxyy, axis=(1,2))
    Ey2_sum = np.sum(E2y * dxxyy, axis=(1,2))
    Ez2_sum = np.sum(E2z * dxxyy, axis=(1,2))
    E2_sum = Ex2_sum + Ey2_sum + Ez2_sum

    Ex_frac = Ex2_sum/E2_sum
    Ey_frac = Ey2_sum/E2_sum
    Ez_frac = Ez2_sum/E2_sum

    TE_frac = (Ex2_sum + Ey2_sum)/E2_sum

    #--- Mode Field Diameter (ISO Standard 11146, D4sig method)
    It = np.sum(E2t * dxxyy, axis=(1,2))
    w_x = 4 * (np.sum(xx_m**2 * E2t * dxxyy, axis=(1,2))/It)**0.5
    w_y = 4 * (np.sum(yy_m**2 * E2t * dxxyy, axis=(1,2))/It)**0.5

    #--- Find Ex Mode
    try:
        is_x_polarized = (Ex_frac >= Ey_frac)
        Ex_mode = {"mode":np.lexsort([w_x**2+w_y**2])[is_x_polarized][0]}
    except:
        Ex_mode = None

    #--- Find Ey Mode
    try:
        is_y_polarized = (Ey_frac > Ex_frac)
        Ey_mode = {"mode":np.lexsort([w_x**2+w_y**2])[is_y_polarized][0]}
    except:
        Ey_mode = None

    #--- Calculate Mode Properties
    mode_params = (frq, width, height)
    eps = eps_function(x_m, y_m)
    if len(eps.shape) > 2:
        epsxx, epsxy, epsyx, epsyy, epszz = np.split(eps, 5, axis=2)
        epsxx = epsxx[:,:,0]
        epsyy = epsyy[:,:,0]
        epszz = epszz[:,:,0]
        # note: epsxy and epsxy have no effect
    else: # isotropic material
        epsxx = eps
        epsyy = eps
        epszz = eps

    for mode in (Ex_mode, Ey_mode):
        if mode == None:
            continue
        idx = mode['mode']

        mode["Ex_frac"] = Ex_frac[idx]
        mode["Ey_frac"] = Ey_frac[idx]
        mode["n_eff"] = n_eff[idx]

        is_core = (np.abs(xx_m) <= width/2.) & (np.abs(yy_m) <= height/2.)

        mode['confinement'] = np.sum((E2t[idx]*dxxyy)[is_core])/It[idx]

        a_eff_num = np.sum(((epsxx*E2x[idx] + epsyy*E2y[idx] + epszz*E2z[idx])*dxxyy))**2
        a_eff_den = np.sum((((epsxx*E2x[idx])**2 + (epsyy*E2y[idx])**2 + (epszz*E2z[idx])**2)*dxxyy)[is_core])
        mode['A_eff'] = a_eff_num/a_eff_den

    #--- Plot Mode Profiles
    if plot_directory is not None:
        #--- Plot Directory
        if not os.path.exists(plot_directory):
            os.mkdir(plot_directory)

        #--- Magnetic Field
        Hx = np.array([mode.get_field("Hx", x=x_m, y=y_m) for mode in modes])
        Hy = np.array([mode.get_field("Hy", x=x_m, y=y_m) for mode in modes])
        Hz = np.array([mode.get_field("Hz", x=x_m, y=y_m) for mode in modes])

        H2x = Hx.real**2 + Hx.imag**2
        H2y = Hy.real**2 + Hy.imag**2
        H2z = Hz.real**2 + Hz.imag**2

        Hx2_sum = np.sum(H2x * dxxyy, axis=(1,2))
        Hy2_sum = np.sum(H2y * dxxyy, axis=(1,2))
        Hz2_sum = np.sum(H2z * dxxyy, axis=(1,2))
        H2_sum = Hx2_sum + Hy2_sum + Hz2_sum

        Hx_frac = Hx2_sum/H2_sum
        Hy_frac = Hy2_sum/H2_sum
        Hz_frac = Hz2_sum/H2_sum

        #TM_frac = (Hx2_sum + Hy2_sum)/H2_sum

        #--- Figure
        extent = (-width*1e6, +width*1e6, -height*1e6, +height*1e6)

        plt.style.use('dark_background')
        c_map = colorcet.cm.bkr

        fig, axs = plt.subplots(
            neigs, 4, figsize=(3*4, 3*neigs*extent[3]/extent[1]))

        #--- Plot Modes
        for idx, axrow in enumerate(axs):
            #--- Ex
            Ex_angle = np.median(np.angle(Ex[idx]) % pi)
            Ex_mag = (Ex[idx] * np.exp(-1j*Ex_angle)).real

            #--- Ey
            Ey_angle = np.median(np.angle(Ey[idx]) % pi)
            Ey_mag = (Ey[idx] * np.exp(-1j*Ey_angle)).real

            #--- Ez
            Ez_angle = np.median(np.angle(Ez[idx]) % pi)
            Ez_mag = (Ez[idx] * np.exp(-1j*Ez_angle)).real

            if np.sum((Ex_mag + Ey_mag + Ez_mag)*dxxyy) < 0:
                Ex_mag *= -1
                Ey_mag *= -1
                Ez_mag *= -1

            #--- Plot Modes
            if Ex_frac[idx] > Ey_frac[idx]:
                clr = 'lightcoral'
                sgn = +1
            else:
                clr = 'lightblue'
                sgn = -1

            c_lim = np.max(E2t[idx])
            axrow[0].pcolormesh(xx*1e6, yy*1e6, sgn*E2t[idx], vmin=-c_lim, vmax=c_lim, cmap=c_map)

            c_lim = np.max(np.abs([Ex_mag, Ey_mag, Ez_mag]))
            axrow[1].pcolormesh(xx*1e6, yy*1e6, sgn*Ex_mag, vmin=-c_lim, vmax=c_lim, cmap=c_map)
            axrow[2].pcolormesh(xx*1e6, yy*1e6, sgn*Ey_mag, vmin=-c_lim, vmax=c_lim, cmap=c_map)
            axrow[3].pcolormesh(xx*1e6, yy*1e6, sgn*Ez_mag, vmin=-c_lim, vmax=c_lim, cmap=c_map)

            #--- Format Axes
            axrow[0].set_title(r'{n_eff}:{:.4g}, D4$\sigma$:({:.2g},{:.2g})$\mu$m'.format(n_eff[idx], w_x[idx]*1e6, w_y[idx]*1e6, n_eff=r"n$_{eff}$"), color=clr, size='medium')
            axrow[1].set_title(r'Ex {:d}${deg}$, E:{:.1%}, M:{:.1%}'.format(int(round(Ex_angle*180/pi)), Ex_frac[idx], Hx_frac[idx], deg=r"^{\circ}"), size='medium')
            axrow[2].set_title(r'Ey {:d}${deg}$, E:{:.1%}, M:{:.1%}'.format(int(round(Ey_angle*180/pi)), Ey_frac[idx], Hy_frac[idx], deg=r"^{\circ}"), size='medium')
            axrow[3].set_title(r'Ez {:d}${deg}$, E:{:.1%}, M:{:.1%}'.format(int(round(Ez_angle*180/pi)), Ez_frac[idx], Hz_frac[idx], deg=r"^{\circ}"), size='medium')

            for ax in axrow:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                ax.set_aspect('equal', 'box')
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontsize("small")
                drawRectangle(ax, left=-0.5*width*1e6, bottom=-height/2.*1e6, width=width*1e6, height=height*1e6, alpha=0.3)

            # Left column
            for ax in axs[:, 0]:
                ax.set_ylabel(r'y ($\mu$m)', size="small")

            # Bottom row
            for ax in axs[-1, :]:
                ax.set_xlabel(r'x ($\mu$m)', size="small")

        fig.suptitle('Height: {:.1f}nm, Width: {:.1f}nm, Wavelength: {:.1f}nm'.format(height*1e9, width*1e9, wl*1e9), y=1., size="large")
        plt.tight_layout()
        file_name = os.path.join(plot_directory,'Modes_height-{:d}nm_width-{:d}nm_wvl-{:d}nm.png'.format(int(round(height*1e9)), int(round(width*1e9)), int(round(wl*1e9))))
        fig.savefig(file_name, dpi=200)
        plt.close()

    #--- Return Result
    return mode_params, n_eff, TE_frac, modes, Ex_mode, Ey_mode

def drawRectangle(
    ax, left, bottom, width, height, fill=False, color='w', **kwargs):
    ax.add_patch(
        plt.matplotlib.patches.Rectangle(
            (left, bottom), width, height,
            fill=fill, color=color, **kwargs))

def make_grid(width, thick, wvl, splitting=10, decay=1e-6, n=1.):
    assert isinstance(splitting, int)
    #--- Fine Grid
    x_fine_grid = np.linspace(-width, +width, num=2*splitting+1)
    y_fine_grid = np.linspace(-thick, +thick, num=2*splitting+1)

    #--- Coarse Grid
    k = 2*pi*np.min(n)/np.max(wvl)
    max_width = width+np.abs(np.log(decay)/k)
    max_thick = thick+np.abs(np.log(decay)/k)

    x_coarse_grid = np.exp(
        np.linspace(np.log(width), np.log(max_width), num=splitting+1))[1:]
    y_coarse_grid = np.exp(
        np.linspace(np.log(thick), np.log(max_thick), num=splitting+1))[1:]

    #--- Grid
    x_grid = np.concatenate((-x_coarse_grid[::-1], x_fine_grid, x_coarse_grid))
    y_grid = np.concatenate((-y_coarse_grid[::-1], y_fine_grid, y_coarse_grid))

    return x_grid, y_grid


#%% Parameters ================================================================

#--- Save Directory
modes_directory = 'Modes'
if not os.path.exists(modes_directory):
    os.mkdir(modes_directory)
file_prefix = 'modes_'
print('prefix: %s'%file_prefix)

#--- Solver Parameters
make_plots = False # set True to save mode field plots (slow)
parallel = False #... doesn't work well

neigs    = 4 # number of eigenmodes to find
eigen_tol = 0 # relative error tolerance, 0 = machine precision (see EMpy)
boundary = '0000' # boundary symmetry, "0000" = no symmetry (see EMpy)

#--- Waveguide Parameters
n0func = SiO2# cladding
n1func = SiO2 # substrate
nfunc = SiN # waveguide

#--- Grid Parameters
grid_scale = 20 # must an integer for the grid to line up with the waveguide edge
box_scale = 1e-3 # sized such that the evanescent decay reaches this level

#--- Frequencies
frequencies = np.linspace(c/1610e-9, c/1510e-9, 50)
#frequencies = np.linspace(c/2500e-9, c/400e-9, 170)

# frequencies = [c/1550e-9]
#wavelengths = c/frequencies
wavelengths = np.array([1550e-9])

#--- Heights - Y
# heights = np.linspace(760e-9, 840e-9, 5)
# heights = [800e-9]
heights = np.array([599e-9, 600e-9, 601e-9])

#--- Widths - X
#widths = np.linspace(505e-9, 2005e-9, 76)
# widths = np.linspace(2005e-9, 3005e-9, 10)
#widths = np.linspace(405e-9, 1005e-9, 31)
# widths = [2500e-9]
widths = [1000e-9, 1200e-9, 1400e-9]

equal_dims = [x in widths for x in heights]
assert not any(equal_dims) #do not try to find modes at equal widths and heights!

total_steps = len(heights) * len(widths) * len(frequencies)


#%% Main ======================================================================

def async_callback(async_result):
    #--- Append Result
    results.append(async_result)

    #--- Update Progress
    (frq, w, t) = async_result[0]
    solver_progress(frq, w, t)

def solver_progress(frq, w, t):
    global t_last, n_step
    #--- Parse Result
    wvl=c/frq

    #--- Solver Progress
    n_step += 1
    t_now = time.time()
    t_step = t_now - t_last
    t_tot = t_now - t_start
    t_rem = (total_steps-n_step) * t_tot/n_step
    t_last = t_now

    #--- Display Progress
    progress = []
    progress.append("{:d}/{:d}:".format(n_step, total_steps))
    progress.append("h={: <4.4g}nm".format(1e9*t))
    progress.append("w={: <4.4g}nm".format(1e9*w))
    progress.append("wl={: <4.4g}nm".format(1e9*wvl))
    progress.append("t_step={: <2.2g}s".format(t_step))
    progress.append("t_tot={:.1f}s".format(t_tot))
    progress.append("t_rem={:.1f}s".format(t_rem))
    print(*progress, sep='\t')

def main():
    global results, t_start, t_last, n_step

    print(f'n_0(SiN) = {SiN(1560)}')

    #--- Output File
    date_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = file_prefix + date_string + '.txt'
    with open(os.path.join(modes_directory,filename), 'w') as outfile:
        #--- Plot Directory
        if make_plots:
            plot_directory = os.path.join(modes_directory,'plots', file_prefix + date_string)
            if not os.path.exists(os.path.join(modes_directory,'plots')):
                os.mkdir(os.path.join(modes_directory,'plots'))
            print(plot_directory)
        else:
            plot_directory = None

        #--- File Header
        column_headers = [
            'height','width','frequency',
            'Ex_n_eff\t','Ex_frac','Ex_core','Ex_A_eff\t',
            'Ey_n_eff\t','Ey_frac','Ey_core','Ey_A_eff']
        print('Effective Refractive Indices:', file=outfile)
        print('rel_box_size:     {:.2g}'.format(box_scale), file=outfile)
        print('rel_grid_spacing: {:.2g}'.format(grid_scale), file=outfile)
        print('rel_tolerance:    {:.2g}'.format(eigen_tol), file=outfile)
        print('eigenvectors:     {:d}'.format(neigs), file=outfile)
        print('boundary cond:    {:s}'.format(boundary), file=outfile)
        print('Wavelengths from {:.3g} um to {:.3g} um in {:d} steps'.format(1e6*np.min(wavelengths), 1e6*np.max(wavelengths), len(wavelengths)), file=outfile)
        print('Heights from {:.3g} um to {:.3g} um in {:d} steps'.format(1e6*np.min(heights), 1e6*np.max(heights), len(heights)), file=outfile)
        print('Widths from {:.3g} um to {:.3g} um in {:d} steps'.format(1e6*np.min(widths), 1e6*np.max(widths), len(widths)), file=outfile)
        print(*column_headers, sep='\t', file=outfile)

        #--- Find Modes
        t_start = t_last = time.time()
        n_step = 0

        if parallel:
            cpu = multiprocessing.cpu_count()
            cpu = 2
            print('multiprocessing with %i cpus'%cpu)
            p = multiprocessing.Pool(cpu)

        for height in heights:
            for width in widths:
                results = []

                #--- Construct Grid
                grid = make_grid(
                    width, height, wavelengths,
                    splitting=grid_scale*1, decay=box_scale, # multiplied grid_scale by 4
                    n=[nfunc(wavelengths), n0func(wavelengths), n1func(wavelengths)])

                #--- Find Modes
                for freq in frequencies:
                    wvl = c/freq
                    if parallel:
                        p.apply_async(
                            rectangular_waveguide_modes,
                            (freq,),
                            dict(
                                neigs=neigs, height=height, width=width,
                                n0=n0func(wvl), n1=n1func(wvl), n2=nfunc(wvl),
                                grid=grid, plot_directory=plot_directory,
                                rel_tol=eigen_tol),
                            callback=async_callback)
                    else:
                        results.append(rectangular_waveguide_modes(
                            *(freq,),
                            **dict(
                                neigs=neigs, height=height, width=width,
                                n0=n0func(wvl), n1=n1func(wvl), n2=nfunc(wvl),
                                grid=grid, plot_directory=plot_directory,
                                rel_tol=eigen_tol)))

                        #--- Solver Progress
                        solver_progress(freq, width, height)

                if parallel:
                    sort_keys = []
                    while len(results)!=len(frequencies):
                        pass # wait for the results to come in
                    for result in results:
                        (frq, w, t), neffs, TE_frac, modes, Ex_mode, Ey_mode = result
                        sort_keys.append([frq, w, t])
                    sort_keys = np.array(sort_keys)
                    # Sort by height, width, and then frequency (t -> w -> frq)
                    sort_ind = np.lexsort(sort_keys.T)
                    results = np.array(results)[sort_ind]

                #--- Parse Results
                for result in results:
                    row_data = []
                    (frq, w, t), neffs, TE_frac, modes, Ex_mode, Ey_mode = result
                    wvl = c/frq
                    row_data.append('{: <6.6g}'.format(t*1e9)) # height
                    row_data.append('{: <6.6g}'.format(w*1e9)) # width
                    row_data.append('{: <6.6g}\t'.format(frq*1e-12)) # frequency
                    if Ex_mode != None:
                        row_data.append('{: <12.12g}'.format(Ex_mode['n_eff']))
                        row_data.append('{: <3.3g}'.format(Ex_mode['Ex_frac']))
                        row_data.append('{: <3.3g}'.format(Ex_mode['confinement']))
                        row_data.append('{: <6.6g}\t'.format(Ex_mode['A_eff']))
                    else:
                        print('No Ex modes found!')
                        row_data.append('{: <12.12g}'.format(np.nan))
                        row_data.append('{: <3.3g}'.format(np.nan))
                        row_data.append('{: <3.3g}'.format(np.nan))
                        row_data.append('{: <6.6g}\t'.format(np.nan))
                    if Ey_mode != None:
                        row_data.append('{: <12.12g}'.format(Ey_mode['n_eff']))
                        row_data.append('{: <3.3g}'.format(Ey_mode['Ey_frac']))
                        row_data.append('{: <3.3g}'.format(Ey_mode['confinement']))
                        row_data.append('{: <6.6g}'.format(Ey_mode['A_eff']))
                    else:
                        print('No Ey modes found!')
                        row_data.append('{: <12.12g}'.format(np.nan))
                        row_data.append('{: <3.3g}'.format(np.nan))
                        row_data.append('{: <3.3g}'.format(np.nan))
                        row_data.append('{: <6.6g}'.format(np.nan))

                    #--- Save Row
                    print(*row_data, sep='\t', file=outfile)

        if parallel:
            p.close()
            p.join()

if __name__=="__main__":
    main()
#%% Plotting single mode profile (TE)
    
#plt.figure('Mode Profile')
#ax = plt.gca()
#ax.set_xlim(extent[0], extent[1])
#ax.set_ylim(extent[2], extent[3])
#ax.set_aspect('equal', 'box')
#for label in ax.get_xticklabels() + ax.get_yticklabels():
#    label.set_fontsize("small")
#drawRectangle(ax, left=-0.5*width*1e6, bottom=-height/2.*1e6, width=width*1e6, height=height*1e6, alpha=0.5)
#c_lim = np.max(E2t[idx])
#ax.pcolormesh(xx*1e6, yy*1e6, sgn*E2t[0], vmin=np.min(sgn*E2t[0]), vmax=c_lim)
#ax.set_xlabel(r'x ($\mu$m)')
#ax.set_ylabel(r'y ($\mu$m)')
#plt.tight_layout()