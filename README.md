# Mode Calculations and Soliton Propagation through Waveguide to read out Squeezing
This script serves the purpose to take dimensions of a waveguide and use them to generate a mode file.
In generating the mode file, I calculate the effective index and nonlinear parameter $\gamma$ for given wavelengths.

### Mode Calculations:
_Fill this in_

### Single Propagation:
_Fill this in_

These output were verified with the Octave Photonics waveguide propagation GUI and with Thomas Charland's MatLab code. 

These outputs have been verified with the experiment and match pretty well.
_To do:_ add this info here

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
13. Calculate the dB of squeezing using $$S_{dB} = -10\log_{10}\left(\frac{\sigma^2_{meas}}{\bar{\sigma^2_{ref}}}\right)$$.
