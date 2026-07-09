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
4. Create vacuum noise and use it to obtain the overlaps `c_vac` which are defined as $$ c_{vac} = \sum \delta\alpha_{\vac}\alpha^*_{LO} $$
  matching the noise readout of balanced homodyne detection.
6. j
7. h
8. h
9. h
