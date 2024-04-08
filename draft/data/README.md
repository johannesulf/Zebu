Supplementary material to DESI's publication "Systematic effects in galaxy-galaxy lensing with DESI" to comply with the data management plan. In the following, we list the files associated with each figure. "title_{a,b}.csv" means that both "title_a.csv" and "title_b.csv" would belong to that figure.

* Figure 1: gravitational_{gt,ds}_des.csv
* Figure 2: ptcl_shear_ratio.csv
* Figure 3: shear_bias_ds_{des,hsc}.csv
* Figure 4: boost_gt_{des,hsc}.csv
* Figure 5: intrinsic_alignment_ds_{des,hsc,kids}.csv
* Figure 6: fiber_assignment_no_iip_ds_hsc.csv
* Figure 7: fiber_assignment_ds_hsc.csv
* Figure 8: lens_magnification_ds_{des,hsc,kids}.csv
* Figure 9: lens_magnification_gt_des.csv
* Figure 10: source_magnification_ds_hsc.csv
* Figure 11: clustering_des.csv
* Figure 12: reduced_shear_ds_hsc.csv
* Figure 13: significance_{gt,ds}_{des,hsc,kids}_{boost,fiber_assignment_no_iip,intrinsic_alignment,lens_magnification,reduced_shear}.csv

For the file names, "gt" indicates tangential shear, $\gamma_{\mathrm{t}}$, measurements whereas "ds" refers to excess surface density, $\Delta\Sigma$, measurements. Except for Figure 2, Figure 11, and Figure 13, the data for all figures is in the form of tables with the following columns.

* bin: A simple index with no special meaning.
* lens_bin: The DESI lens bin. We use the convention that BGS-1, BGS-2, BGS-3, LRG-1, and LRG-2 correspond to the lens bins 0, 1, 2, 3, and 4, respectively. Lens bin 5 would correspond to LRG-3 which was not included in this analysis.
* source_bin: The source tomographic bin, e.g., source bin 0 would correspond to HSC-1, DES-1, or KiDS-1 in the figures, depending on the lensing survey.
* radial_bin: The radial bin, i.e., the bin in $\theta$ or $r_{\mathrm{p}}$, depending on whether the tangential shear $\gamma_{\mathrm{t}}$ or the excess surface density $\Delta\Sigma$ is calculated.
* r: The radial distance in $\theta$ (in arcmin) or $r_{\mathrm{p}}$ (in $\mathrm{Mpc}/h$).
* value: The absolute value of $\gamma_{\mathrm{t}}$ or $\Delta\Sigma$ for Figure 1 and the relative size of the effect in percent for all other figures.
* error: Uncertainty on the value.