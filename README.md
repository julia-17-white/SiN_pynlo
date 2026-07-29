# Mode Calculations and Soliton Propagation through Waveguide to read out Squeezing
This script serves the purpose to take dimensions of a waveguide and use them to generate a mode file.
In generating the mode file, I calculate the effective index and nonlinear parameter $\gamma$ for given wavelengths.

### Fiber Simulation:
This script takes the soliton out of the oscillator in our Menhir comb and applies the nonlinear effects of the PM-1550 in the laser and that couples out of the laser as well as the ND fiber that is used to compress the pulse back to being near the foueier-transform limit. This is done in `fiber_sim`'s `nd_run()` method and the results are a close match to the laser spectrum we see out of these fibers. The results are also verified to match the results in the GUI written by Xiangwen Gao for all fiber simulation in our group.

### Mode Calculations:
These use [EMpy](https://github.com/lbolla/EMpy/tree/master) and were provided by Abijith Kowligy. 
_Fill this in and understand these more_

### Single Propagation:
The waveguide mode is created from the mode file generated in `mode_match.py` and the pulse is the output from the `fiber_sim.nd_run()` method. From the mode file, I create a gamma and n_eff spline. The indices of refraction are used to calculate the dispersion `beta_v` and the nonlinearity gamma is used to calculate my third order nonlinear coupling `g3_v`. These then create the PyNLO mode and using `sim.simulate()`, I can propagate the pulse through this mode. There are a few methods that allow me to further analyze this output pulse:
* `nice_plot()` produces the standard time and frequency domain plots of the input and output pulse as well as its propagation through the waveguide.
* `plot_osa_spectrum()` produces a plot of the input and output pulses in dBm/nm and over wavelength. This matches the standard OSA readout.
* `pulse_interference()` produces a plot of the two pulses used as parameters as well as their interference based on addition.


I verified my PyNLO outputs with the Octave Photonics waveguide propagation GUI and with Thomas Charland's MatLab code. These outputs also match the spectral outputs I see from my waveugides quite well.


### Noise Injection:
[Paschotta's Noise of MLLs Part 1](https://link.springer.com/article/10.1007/s00340-004-1547-x) discusses that my frequency resolution should be $\delta v = 1/T$ where $T$ is the temporal range. By default pynlo's `dv` is the "frequency step size. This is equal to the reciprocal of the time window." This matches the Paschotta definition. Pynlo also uses `a_v` which is the "root power spectrum" so the total energy is $E = \sum \left|a_v\right|^2 \text{d}v$. This means that I can find the noise amplitude for a given pulse by with $a_{v,noise} = \sqrt{\frac{.5 h v}{\text{d}v}}$. This noise is then multiplied by some injected complex noise and added to the original pulse so it workes as the magnitude of the noise added to the pulse.

The injected complex noise is found by generating two sets of Gaussian noise with a mean of zero and a standard deviation of one. These are then cast together as complex noise using $n = \frac{1}{\sqrt{2}}\left(n_{real} + i n_{complex}\right)$. This complex noise is then multiplied by the noise amplitude for the given pulse $a_{v,noise}$ and added to the original pulse.

I believe that I shouldn't need to implement a proper Poissonian distribution here because my Normal distribution has `len(v_grid)` which is the length of the PyNLO pulse's frequency grid which has $N \approx 8000$.

### Squeezing Search:
This squeezing search occurs in a number of steps.
1. Create a base waveguide and pulse using the single propagation work. This uses options where none of the plots are generated.
2. Create an un-propagagated LO pulse to replicate the delay line pulse. This pulse is called `_worker_a_v_lo`.
3. Create an noiseless pulse that is propagated through the waveguide. This pulse is called `_worker_a_v_clean`.
4. Measure the intensity of the signal pulse that was propagated with noise. This is `I_ref` and it is defined as $I_{ref} = \sum \left|a_{sqz}\right|^2 \text{d}v$ where $a_{sqz}$ is the signal pulse with its noise.
5. Scan through various $\theta$ values and create an auxillary (LO) pulse `aux` which is defined as $a_{lo} = \mu a_{clean} e^{i\theta}$ where $\mu$ is a ratio intended to ensure a 100:1 power split between the signal and auxillary pulses and $a_{clean}$ is the un-propagated LO pulse.
6. Measure the intensity of the signal pulse that was propagated with noise interfered with the LO puslse. This is `I` and it is defined as $I_{sig} = \sum \left|a_{sqz} + a_{lo} * 0.97\right|^2 \text{d}v$. The factor of $0.97$ is included to account for inperfect mode overlap between the pulses.
7. I use `np.var` to calculate the variances of these intensities. This uses the standard variance formula $$\frac{\sum_i|a_i-\bar{a}|}{N}$$. `var_signal` is the signal's noise variance and `var_vacuum` is the variance of the vacuum noise.
8. I iincorporate the $\eta = 0.78$ detection loss observed in Dan and Molly-Kate's paper. This is done as $\sigma^2_{meas} = \eta \sigma^2_{sig} + \left(1-\eta\right)\sigma^2_{ref}$.
13. Calculate the dB of squeezing using $$S_{dB} = 10\log_{10}\left(\frac{\sigma^2_{meas}}{\bar{\sigma^2_{ref}}}\right)$$.
