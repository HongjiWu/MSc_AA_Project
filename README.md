# MSc_AA_Project
repo for master project: Assessing the Practical Risks of Text Authorship Attribution Attacks

All Dataset should be stored in /data folder, however, since the size of it exceed github limit
you would need to download it in the following drive:


All experiment output is stored in /output

For running Exp.1 Varying Number of Training Samples, run exp_tr_samples.py
For running Exp.2 Varying Number of Candidate Authors, run exp_author_num.py
For runnning Exp.3 Varying Length of Data Samples, run exp_sample_length.py
For running Exp.4 Varying Content Divergence between Referencing and Targetting Data Samples, run exp_sim.py

For adding more aa_method in experiment, you could follow the guideline in /authorship_attribution/methods/base_aa_method.py
and add your new methods in /authorship_attribution/methods similar to the exising one

For tuning the hyperparameter of aa_method, run hyperparm_tune.py

There is a jupyter notebook used to generate all the plots in report & ppt in /output
and all the plots generated is in /output/fig
