FILES := \
	stacks/plots_absolute/gravitational_gt_des \
	stacks/plots_absolute/gravitational_ds_des \
	misc/particles/ptcl_shear_ratio \
	stacks/plots_relative/shear_bias_ds_des \
	stacks/plots_relative/shear_bias_ds_hsc \
	stacks/plots_relative/boost_gt_des \
	stacks/plots_relative/boost_gt_hsc \
	stacks/plots_relative/intrinsic_alignment_ds_des \
	stacks/plots_relative/intrinsic_alignment_ds_hsc \
	stacks/plots_relative/intrinsic_alignment_ds_kids \
	stacks/plots_relative/fiber_assignment_no_iip_ds_hsc \
	stacks/plots_relative/fiber_assignment_ds_hsc \
	stacks/plots_relative/lens_magnification_ds_des \
	stacks/plots_relative/lens_magnification_ds_hsc \
	stacks/plots_relative/lens_magnification_ds_kids \
	stacks/plots_relative/lens_magnification_gt_des \
	stacks/plots_relative/source_magnification_ds_hsc \
	stacks/plots_relative/reduced_shear_ds_hsc \
	misc/blending/clustering_des \
	misc/significance/significance

all: $(FILES)
	mkauthlist -f -j mnras -c "" author_list.csv tmp.tex
	grep '\$$\^{[0-9,]*}\$$$$' tmp.tex > authors.tex
	grep '^\$$\^{[0-9,]*}\$$' tmp.tex > affiliations.tex
	rm tmp.tex

	latexdiff --disable-citation-markup ms_cwr.tex ms.tex > diff_cwr.tex
	bibm pdflatex diff_cwr.tex

	bibm pdflatex ms.tex
	zip -FS source.zip ms.tex authors.tex affiliations.tex ms.bbl bibliography.bib *.pdf -x diff_*.pdf

$(FILES):
	mkdir -p plots
	mkdir -p data
	cp ../$@.pdf plots
	cp ../$@*.csv data

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
