# HeII_input_file_maker.py

__HeII_input_file_maker.py__ creates a reionization history table with desired parameters of HeII reionization. 

To run __HeII_input_file_maker.py__, the parameters of HeII reionization must be input as command line arguments. These parameters are:

__(1)__ Spectral index of quasars. The recommended range for this parameter is 1.1-2.0.

__(2)__ The threshold energy that separates long-mean-free-path photons (photons that heat the IGM uniformly) from short-mean-free-path photons (photons that contribute to the creation of HeIII bubbles) in electron volts. The thermal history is weakly dependent on this parameter, but recommended values are ~100-200.

__(3)__ Duration and timing of HeII reionization. The options are 'quasar' and 'linear'. The 'quasar' option uses a quasar emissivity function to determine the reionization history. The quasar emissivity histories are from Khaire et al. (2015) and Haardt and Madau (2012). The default is Khaire et al. (2015). The 'linear' option allows the user to select the starting and ending redshift of HeII reionization. The HeIII fraction will be a linear function in redshift between these two redshifts. This parameter is optional and will default to 'quasar' if not provided.

__(4)__ If 'linear', the starting HeII reionization redshift. Only to be used when (3) is 'linear'. 

__(5)__ If 'linear', the ending HeII reionization redshift. HeII reionization is observed to end at z ~ 2.8. Only to be used when (3) is 'linear'. 

__The following examples provide recommended parameters:__

_'Quasar' mode:_

`python HeII_input_file_maker.py 1.7 150.0` 

_'Linear' mode:_

`python HeII_input_file_maker.py 1.7 150.0 'linear' 4.0 2.8`

Note, the parameters used will be printed in a comment at the top of the output table.
