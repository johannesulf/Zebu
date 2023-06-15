all:
	cp ../stacks/plots_absolute/gravitational_gt.pdf ./
	cp ../stacks/plots_absolute/gravitational_ds.pdf ./
	cp ../stacks/plots_relative/boost_gt.pdf ./
	cp ../stacks/plots_relative/intrinsic_alignment_ds.pdf ./
	cp ../stacks/plots_relative/intrinsic_alignment_ds.pdf ./fiber_incompleteness_ds.pdf
	cp ../stacks/plots_relative/lens_magnification_ds.pdf ./
	cp ../stacks/plots_relative/source_magnification_ds.pdf ./
	
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

