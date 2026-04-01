We are trying to find templates that can describe the spectra. 

The model for the spectra is that

prediction_if = sum_k alpha_ik t_kf(1+z_i) + noise

where i is the galaxy index (Nd) and f is the frequency index (Nf) and k is the template index (Nt). Nf is O(1e4), Nt is a few (say 10 max). 
t_kf=(1+z_i) means template k for which the wavelenghts have been shifted by (1+z_i).
In general we don't know z_i and when searchin for templates we will simply marginalise over zs and over possible alphas.

We will additionally, at the same time for for distribution of galaxies n(z) binned in some bins.

So, the loss from galaxy goodness of fit for a single galaxy is

p_i = sum_z exp(-0.5*sum_f ( (data_if-prediction_if)**2*(inverse_variance_if))
                -0.5*(z-z_prior_i)**2/z_err_i**2) * n(z)

and the total loss is

loss = sum_i log p_i

Later z_err might blow up for the prior to be essentially un-informative.

We will use a combination of jax and/or pytorch to do the fitting using modern ML techniques on a GPU.

The basic parameters are zlow, zhigh and Nz specifyin the size of the redshift search grid.

In the data (see py/spectra_loader.py), wave array is fixed to be between wave_min=3600A and wave_max= 9824A.

The template is parameterized as Nt x Nft  array, where there are Nft wavelength points between wave_min/(1+zmax) and wave_max/(1+zmin) (so the template always covers the data when shifted by (1+z)).

You should precompute the weights used to interpolate to observed wave-grid for any z in the zgrid.

nz should be an array that spans zmin to zmax, but with much less resolution, say Nnz=50.

Then in the optimization loop you do the following:

 - normalize nz array so that it adds up to unity
 - Outer loop is over zs in zgrid. For each z:
     - interpolate templates onto wave
     - find optimal alphas given the current templates using the right jnp.linalg.solve to keep stuff differential
     - calculate contribution to p_i at current z using equation above and add up to an array over galaxies
  - Now that you have p_is, take log and sum and this is your loss
  - update template arrays and n(z).
  -repeat

You should write a code that does all this and is efficiently parallelized. Feel free to suggest further optimization or ideas for increased speed.

The trainig routine should present itself as a CLI utility with command line options to control parameters.  For testing it should allow loading a few hundred galaxies of a given target_id and with certain zmin and zmax. 

You should also write a jupyter notebook that should use this infrastructure, to:
 - load the best parameters
 - find best fit alpha and zs for a selection of galaxies and plot those
 - plot n(z) and compare with true n(z) of the training dataset.

Modules for the code should live in py/ directory. Training utilities should live in scripts/ . Notebooks should live in notebooks.




     
      





