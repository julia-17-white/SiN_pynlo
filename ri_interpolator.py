#%% Imports ===================================================================

import numpy as np
from scipy import constants, interpolate, ndimage
from scipy.interpolate import RegularGridInterpolator
import pickle, os, time

from matplotlib import widgets as mpl_widgets
import matplotlib.pyplot as plt
import colorcet
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'

c = constants.c
pi = constants.pi


#%% Parameters ================================================================

#--- Interpolation
interp_ord = 3

#--- Modes
filename = "modes_2026-04-14_14-30-23.txt" # full range SiN data

skip_header = 10

#--- Plot Parameters
material = "SiN" # Just a prefix for plot titles and the output files
initial_thick = 599 #690 # nm, 800 nm or 420 nm, 650, 690 or 350
initial_pump = 1560 # nm
n2=2.3e-19 # SiN nonlinearity

'''
Yu M, "Coherent two-octave-spanning supercontinuum generation in lithium-niobate waveguides", OL (2019).
n2 = 2.5e-19 m^2/W
strong 2nd order nonlinearity, r33  3 × 10−11 m∕V
'''
num_interp = int(1e3)
gui_widgets = {}
im = {}
lines = []

#--- Load Data
filename = os.path.join('Modes',filename)
if not os.path.exists(filename+'.pickle'):
    #--- Load Data
    print('loading files')
    data = np.loadtxt(filename, dtype=float, unpack=True, skiprows=skip_header)

    #--- Sample Space
    all_sim_thicks = data[0]*1e-9 # nm to m
    all_sim_widths = data[1]*1e-9 # nm to m
    all_sim_freqs = data[2]*1e12 # THz to Hz

    sim_thicks = np.unique(all_sim_thicks)
    sim_widths = np.unique(all_sim_widths)
    sim_freqs  = np.unique(all_sim_freqs)

    sim_thicks_3D, sim_widths_3D, sim_freqs_3D = np.meshgrid(sim_thicks, sim_widths, sim_freqs, indexing='ij')

    print("X dims:{:}".format(sim_widths))
    print("Y dims:{:}".format(sim_thicks))

    #--- Ex
    all_sim_Ex_neff = data[3]
    all_sim_Ex_frac = data[4]
    all_sim_Ex_confinement = data[5]
    all_sim_Ex_A_eff = data[6]

    #--- Ey
    all_Ey_neff = data[7]
    all_sim_Ey_frac = data[8]
    all_sim_Ey_confinement = data[9]
    all_sim_Ey_A_eff = data[10]

    #--- Grid Data
    sim_Ex_neffs_3D = np.empty((sim_thicks.shape[0], sim_widths.shape[0], sim_freqs.shape[0]))
    sim_Ey_neffs_3D = np.empty((sim_thicks.shape[0], sim_widths.shape[0], sim_freqs.shape[0]))
    sim_Ex_aeffs_3D = np.empty((sim_thicks.shape[0], sim_widths.shape[0], sim_freqs.shape[0]))
    sim_Ey_aeffs_3D = np.empty((sim_thicks.shape[0], sim_widths.shape[0], sim_freqs.shape[0]))

    for i,thick in enumerate(sim_thicks):
        for j,width in enumerate(sim_widths):
            for k,freq in enumerate(sim_freqs):
                is_twl = ((all_sim_thicks==thick) & (all_sim_widths==width) & (all_sim_freqs==freq))
                # Ex Polarized
                # sim_Ex_neffs_3D[i,j,k] = np.real(all_sim_Ex_neff[is_twl])
                # sim_Ex_aeffs_3D[i,j,k] = np.real(all_sim_Ex_A_eff[is_twl])
                sim_Ex_neffs_3D[i,j,k] = np.real(all_sim_Ex_neff[is_twl][0])
                sim_Ex_aeffs_3D[i,j,k] = all_sim_Ex_A_eff[is_twl][0]
                # Ey Polarized
                # Add the [0] index to these lines as well
                sim_Ey_neffs_3D[i,j,k] = np.real(all_Ey_neff[is_twl][0])
                sim_Ey_aeffs_3D[i,j,k] = np.real(all_sim_Ey_A_eff[is_twl][0])

    #--- Save to Pickle
    with open(filename+'.pickle', 'wb') as outfile:
        data = (
            sim_thicks, sim_widths, sim_freqs,
            sim_thicks_3D, sim_widths_3D, sim_freqs_3D,
            sim_Ex_neffs_3D, sim_Ex_aeffs_3D,
            sim_Ey_neffs_3D, sim_Ey_aeffs_3D)
        pickle.dump(data, outfile)

else: #TODO: save to compressed npz file instead
    #--- Load Data
    print('loading pickle')
    with open(filename+'.pickle', 'rb') as infile:
        data = pickle.load(infile, fix_imports=True, encoding='bytes')
    #--- Sample Space
    sim_thicks = data[0]
    sim_widths = data[1]
    sim_freqs = data[2]

    sim_thicks_3D = data[3]
    sim_widths_3D = data[4]
    sim_freqs_3D = data[5]

    #--- Ex Mode
    sim_Ex_neffs_3D = data[6]
    sim_Ex_aeffs_3D = data[7]

    #--- Ey Mode
    sim_Ey_neffs_3D = data[8]
    sim_Ey_aeffs_3D = data[9]

sim_waves = c/sim_freqs

sim_thick_step = np.diff(np.sort(sim_thicks)).mean()
sim_width_step = np.diff(np.sort(sim_widths)).mean()
sim_freq_step  = np.diff(np.sort(sim_freqs)).mean()

#%% Methods ===================================================================
def refractive_index_and_gamma(
    thicks, widths, freqs, mode='Ex', n2=n2, interpolation_order=3): # interp_order
    """
    """
    #--- Select Mode
    if mode=='Ex':
        sim_neffs_3D = sim_Ex_neffs_3D
        sim_aeffs_3D = sim_Ex_aeffs_3D
    elif mode=='Ey':
        sim_neffs_3D = sim_Ey_neffs_3D
        sim_aeffs_3D = sim_Ey_aeffs_3D
    else:
        raise TypeError("Mode '{:}' is not in ('Ex', 'Ey')".format(mode))

    #--- Interpolation Coordinates
    thicks_3D, widths_3D, freqs_3D  = np.meshgrid(thicks, widths, freqs, indexing='ij')
    coords_3D = (
        (thicks_3D-np.min(sim_thicks))/sim_thick_step,
        (widths_3D-np.min(sim_widths))/sim_width_step,
        (freqs_3D-np.min(sim_freqs))/sim_freq_step)

#    coords_3D = (
#        thicks_3D,
#        (widths_3D-np.min(sim_widths))/sim_width_step,
#        (freqs_3D-np.min(sim_freqs))/sim_freq_step)
    
    #--- Refractive Indices
    neffs = ndimage.interpolation.map_coordinates(
        sim_neffs_3D, coords_3D, mode = 'nearest', order=1)#interpolation_order)

    #--- Effective Areas
    aeffs = ndimage.interpolation.map_coordinates(
        sim_aeffs_3D, coords_3D, mode = 'nearest',order=1)#interpolation_order)

    #--- Nonlinear Parameters
    gs = (2*pi/c)*n2 * freqs_3D/aeffs
    # print(f'gamma = {gs}')

    #--- Return Values
    if len(thicks) > 1:
        thick_sel = slice(None)
    else:
        thick_sel = 0
    if len(widths) > 1:
        width_sel = slice(None)
    else:
        width_sel = 0
    if len(freqs) > 1:
        freq_sel = slice(None)
    else:
        freq_sel = 0

    neffs = neffs[thick_sel, width_sel, freq_sel]
    gs = gs[thick_sel, width_sel, freq_sel]
    aeffs = aeffs[thick_sel, width_sel, freq_sel]# extra added


    return neffs, gs, aeffs

def update_plot(*args):
    '''
    [1] Agrawal, G. P. Nonlinear Fiber Optics. Academic, Oxford, 2013;.
    1.2.3, "Chromatic Dispersion", pg 8
    [2] Agrawal, G. P. Nonlinear Fiber Optics. Academic, Oxford, 2013;.
    12.1.2, "Generation of Dispersive Waves", pg 501
    '''
    t = time.time()
    #--- Update Values
    thickness = 1e-9*gui_widgets["thick"].val
    pump_wvl = 1e-9*gui_widgets["pump"].val
    pump_frq = c/pump_wvl
    mode = gui_widgets["mode"].value_selected

    print(mode, thickness, pump_wvl*1e9)

    #--- Interpolation Coordinates
    widths = np.linspace(sim_widths.min(), sim_widths.max(), sim_widths.size*3)
    waves = np.linspace(sim_waves.min(), sim_waves.max(), num_interp)
    freqs = c/waves

    #--- Simulated RIs and Gammas
    sim_neff, sim_gamma, sim_aeff = refractive_index_and_gamma(
        [thickness], widths, sim_freqs, mode=mode)

    #--- RI Derivatives
    # Plot Grid
    neff = np.empty((len(widths), len(freqs)))
    dndv = np.empty((len(widths), len(freqs)))
    d2ndv2 = np.empty((len(widths), len(freqs)))
    # Pump Frequency
    neff_pump = np.empty((len(widths), 1))
    dndv_pump = np.empty((len(widths), 1))
    d2ndv2_pump = np.empty((len(widths), 1))

    # Spline Interpolation along each Width
    for idx, ry in enumerate(sim_neff):
        interp = interpolate.InterpolatedUnivariateSpline(sim_freqs, ry, k=3)
        # Plot Grid
        neff[idx] = interp(freqs)
        dndv[idx] = interp.derivative(n=1)(freqs)
        d2ndv2[idx] = interp.derivative(n=2)(freqs)
        # Pump Frequency
        neff_pump[idx] = interp(pump_frq)
        dndv_pump[idx] = interp.derivative(n=1)(pump_frq)
        d2ndv2_pump[idx] = interp.derivative(n=2)(pump_frq)

    #--- Nonlinear Parameter
    width_step = np.diff(widths).mean()
    widths_2D, freqs_2D  = np.meshgrid(widths, freqs, indexing='ij')
    coords_2D = (
        (widths_2D-np.min(widths))/width_step,
        (freqs_2D-np.min(sim_freqs))/sim_freq_step)

    gamma = ndimage.interpolation.map_coordinates(
        sim_gamma, coords_2D, order=3)

    #--- Group Index
    n_g = neff + freqs_2D*dndv
    n_g_pump = neff_pump + pump_frq*dndv_pump

    #--- Walk-Off Parameter "d12"
    d12 = (n_g_pump - n_g)/c
    d12 *= 1e15/1e3 #fs/mm

    #--- Dispersion Parameter "D"
    gvd_D = -(freqs_2D/c)**2 * (2*dndv + freqs_2D*d2ndv2)
    gvd_D *= 1e12/(1e9*1e-3) # ps/nm/km

    #--- Dispersive Wave Phase Matching
    # Solitonic Radiation
    beta_pump = 2*pi*pump_frq/c * neff_pump
    beta1_pump = n_g_pump/c

    beta_sol = (beta_pump + 2*pi*(freqs_2D - pump_frq)*beta1_pump)

    # Dispersive Radiation
    beta_DW = 2*pi/c*freqs_2D * neff

    # Phase Mismatch
    mismatch = beta_DW - beta_sol
    DW_Lc = 10*np.log10(0.5*pi/np.abs(mismatch)) # coherence length # last row gives zero for all values, not sure why

    print('calculated everything in %.2f sec'%(time.time()-t))
    t = time.time()

    #--- Update Plots
    im['n'].set_data(neff)
    im['ng'].set_data(n_g)
    im['d12'].set_data(d12)
    im['g'].set_data(gamma)
    im['gvd'].set_data(gvd_D)
    im['dw'].set_data(DW_Lc)

    im['n'].set_clim(neff.min(), neff.max())
    im['ng'].set_clim(n_g.min(), n_g.max())
    im['d12'].set_clim(-100, 100)
    im['g'].set_clim(gamma.min(), gamma.max())
    im['gvd'].set_clim((-gvd_D.max(), gvd_D.max()))
    im['dw'].set_clim((-30, 0))#(DW_Lc.min(), DW_Lc.max()))

    for l in lines:
        l.set_xdata([1e9*pump_wvl])

    im['fig'].suptitle(
        '{:} - {:} nm Thick - {:} Polarized - {:} nm Pump'.format(
            material,
            int(round(1e9*thickness)),
            mode,
            int(round(1e9*pump_wvl))),
        weight='bold', fontsize=16)#, y=.99)

    im['fig'].canvas.draw_idle() # Update the plots.

    print('updated plots in %.3f sec'%(time.time()-t) )


# %% Main =====================================================================

if __name__ == '__main__':

    #--- GUI Setup
    figc = plt.figure("control", figsize=[6, 1], constrained_layout=True)
    gsc = figc.add_gridspec(nrows=2, ncols=3)
    axsc = []
    axsc.append(figc.add_subplot(gsc[0, 1:]))
    axsc.append(figc.add_subplot(gsc[1, 1:]))
    axsc.append(figc.add_subplot(gsc[:2, 0]))

    slider_thick = mpl_widgets.Slider(
        axsc[0], 'Thickness (nm)',
        np.min(sim_thicks*1e9), np.max(sim_thicks*1e9), valinit=initial_thick)    
    slider_pump = mpl_widgets.Slider(
        axsc[1], 'Pump Wavelength (nm)',
        np.min(c/sim_freqs*1e9), np.max(c/sim_freqs*1e9), valinit=initial_pump)
    for slider in (slider_thick, slider_pump):
        slider.on_changed(update_plot)

    radio_mode = mpl_widgets.RadioButtons(
        axsc[2], ['Ex', 'Ey'], active=0)
    radio_mode.on_clicked(update_plot)

    # for circle in radio_mode.circles: # adjust radius here. The default is 0.05
    #     circle.set_radius(0.1)

    gui_widgets["thick"] = slider_thick
    gui_widgets["pump"] = slider_pump
    gui_widgets["mode"] = radio_mode

    #--- Plot Setup
    plt.style.use('dark_background')
    im['fig'] = plt.figure("data", figsize=(14,9), constrained_layout=True)
    gs = im['fig'].add_gridspec(nrows=3, ncols=2)
    axs = []
    axs.append(im['fig'].add_subplot(gs[0, 0]))
    axs.append(im['fig'].add_subplot(gs[1, 0], sharex=axs[0], sharey=axs[0]))
    axs.append(im['fig'].add_subplot(gs[2, 0], sharex=axs[0], sharey=axs[0]))
    axs.append(im['fig'].add_subplot(gs[0, 1], sharex=axs[0], sharey=axs[0]))
    axs.append(im['fig'].add_subplot(gs[1, 1], sharex=axs[0], sharey=axs[0]))
    axs.append(im['fig'].add_subplot(gs[2, 1], sharex=axs[0], sharey=axs[0]))

    plt_extent = (
        c/sim_freqs.max()*1e9, c/sim_freqs.min()*1e9,
        sim_widths.min()*1e9, sim_widths.max()*1e9)

    im['n'] = axs[0].imshow( # refractive index
        [[]], extent=plt_extent, origin='lower', aspect='auto',
        interpolation='bilinear', cmap=colorcet.cm.rainbow_bgyrm_35_85_c71)
    im['ng'] = axs[1].imshow( # group index
        [[]], extent=plt_extent, origin='lower', aspect='auto',
        interpolation='bilinear', cmap=colorcet.cm.rainbow_bgyrm_35_85_c71)
    im['d12'] = axs[2].imshow( # walk-0ff parameter
        [[]], extent=plt_extent, origin='lower', aspect='auto',
        interpolation='bilinear', cmap=colorcet.cm.diverging_bwr_20_95_c54)#colorcet.cm.diverging_bkr_55_10_c35)
    im['g'] = axs[3].imshow( # nonlinear parameter
        [[]], extent=plt_extent, origin='lower', aspect='auto',
        interpolation='bilinear', cmap=colorcet.cm.rainbow_bgyrm_35_85_c71)
    im['gvd'] = axs[4].imshow( # group velocity dispersion
        [[]], extent=plt_extent, origin='lower', aspect='auto',
        interpolation='bilinear', cmap=colorcet.cm.diverging_bwr_20_95_c54)
    im['dw'] = axs[5].imshow( # dispersive wave phase matching
        [[]], extent=plt_extent, origin='lower', aspect='auto',
        interpolation='bilinear', cmap=plt.cm.magma)#, clim=(-30,0))

    cbar_n = im['fig'].colorbar(im['n'], ax=axs[0])
    cbar_ng = im['fig'].colorbar(im['ng'], ax=axs[1])
    cbar_d12 = im['fig'].colorbar(im['d12'], ax=axs[2])
    cbar_g = im['fig'].colorbar(im['g'], ax=axs[3])
    cbar_gvd = im['fig'].colorbar(im['gvd'], ax=axs[4])
    cbar_dw = im['fig'].colorbar(im['dw'], ax=axs[5])

    cbar_n.set_label('Refractive Index')
    cbar_ng.set_label('Group Index')
    cbar_d12.set_label('Walk-Off Parameter (fs/mm)')
    cbar_g.set_label(r'$\gamma$ (1/(W$\cdot$m))')
    cbar_gvd.set_label('GVD (ps/nm/km)')
    cbar_dw.set_label('DW Phase Matching')

#    im['fig'].subplots_adjust(left=0.06, bottom=0.07, right=0.97, top=0.96, wspace=0.06, hspace=0.10)
    axs = np.reshape(axs, (2,3)).T
    for ax in axs.ravel():
        lines.append(ax.axvline(initial_pump, c='w', alpha=0.5))
        ax.grid(True, alpha=.1)

    for ax in axs[:,0]: # left side
        ax.set_ylabel('Waveguide Width (nm)')

    for ax in axs[-1]: # bottom
        ax.set_xlabel('Wavelength (nm)')

    #--- Draw Plots
    update_plot()

    plt.show()
