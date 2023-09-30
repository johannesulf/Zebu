all:
	cp ../stacks/plots_absolute/gravitational_gt_des.pdf ./
	cp ../stacks/plots_absolute/gravitational_ds_des.pdf ./
	cp ../misc/particles/ptcl_shear_ratio.pdf ./
	cp ../stacks/plots_relative/shear_bias_ds_des.pdf ./
	cp ../stacks/plots_relative/shear_bias_ds_hsc.pdf ./
	cp ../stacks/plots_relative/boost_gt_des.pdf ./
	cp ../stacks/plots_relative/boost_gt_hsc.pdf ./
	cp ../stacks/plots_relative/intrinsic_alignment_ds_*.pdf ./
	cp ../stacks/plots_relative/fiber_assignment_no_iip_ds_hsc.pdf ./
	cp ../stacks/plots_relative/fiber_assignment_ds_hsc.pdf ./
	cp ../stacks/plots_relative/lens_magnification_ds_*.pdf ./
	cp ../stacks/plots_relative/lens_magnification_gt_des.pdf ./
	cp ../stacks/plots_relative/source_magnification_ds_hsc.pdf ./
	cp ../stacks/plots_relative/reduced_shear_ds_hsc.pdf ./
	cp ../misc/blending/clustering_des.pdf ./
	cp ../misc/significance/significance.pdf ./
	
	bibm pdflatex ms.tex
	zip source.zip *.tex ms.bbl bibliography.bib *.pdf

clean:
	rm -f *.log
	rm -f *.bbl
	rm -f *.bcf
	rm -f *.blg
	rm -f *.aux
	rm -f *.xml
	rm -f *.out
	rm -f *.pdf
	rm -f *.zip

