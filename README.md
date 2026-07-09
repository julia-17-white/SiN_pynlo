# Mode Calculations and Soliton Propagation through Waveguide to read out Squeezing
This script serves the purpose to take dimensions of a waveguide and use them to generate a mode file.
In generating the mode file, I calculate the effective index and nonlinear parameter $\gamma$ for given wavelengths.

### Mode Calculations:
_Fill this in_

### Single Propagation:
_Fill this in_

These output were verified with the Octave Photonics waveguide propagation GUI and with Thomas Charland's MatLab code. 

*to do:* verify these with waveguides in your experiment.

### Squeezing Search:
This squeezing search occurs in a number of steps.
1. Create a base waveguide and pulse using the single propagation work. This uses options where none of the plots are generated.
2. Create an un-propagagated LO pulse to replicate the delay line pulse. This pulse is called `_worker_a_v_lo`.
3. Create an noiseless pulse that is propagated through the waveguide. This pulse is called `_worker_a_v_clean`.
4. Create vacuum noise and use it to obtain the overlaps `c_vac` which are defined as $$c_{vac} = \sum \delta\alpha_{vac}\alpha^*_{LO}$$
  matching the noise readout of balanced homodyne detection.
6. Create the noisy initial pulse, propagate the pulse through the waveguide, and use the output to obtain the balanced homodyne overlaps `c_sig` of the signal's noise with the magnitude of the LO pulse. These overlaps are defined as $$c_{sig} = \sum \delta\alpha_{sig}\alpha^*_{LO}$$. To compute the signal's noise, I subtract the propagated clean pulse from the propagated noisy pulse.
7. Calculate the quadratures with a phase sweep. In doing this, I use $x_{sig} = c_{sig}e^{-i\theta}$ and $x_{vac} = c_{vac}e^{-i\theta}$ with $0\le \theta < 2\pi$. This is essentially adding a slightly different phase (delay) to each of the propagation outputs to mimic scanning the delay line.
8. I use `np.var` to calculate the variances of these quadratures. This uses the standard variance formula $$\frac{\sum_i|a_i-\bar{a}|}{N}$$. `var_signal` is the signal's noise variance and `var_vacuum` is the variance of the vacuum noise.
9. Calculate the dB of squeezing using $$S_{dB} = 10\log_{10}\left(\frac{\sigma^2_{sig}}{\bar{\sigma^2_{vac}}}\right)$$. The denominator is using the average vacuum or shot noise so that I can see how far above and below it I am when I plug in the maximum and minimum signal noise vairances or plot $S_{dB}$.
