
%.pdf:  %.tex techreportacmart.cls ACM-Reference-Format.bst
	pdflatex $<
	- bibtex $*
	pdflatex $<
	pdflatex $<
	while ( grep -q '^LaTeX Warning: Label(s) may have changed' $*.log) \
	do pdflatex $<; done

sample-acmtog.pdf:

clean:
	@echo "Cleaning..."
	mv sample-acmtog.tex sample_acmtog.tex
	rm -f comment.cut 
	rm -f sample-acmtog.*
	mv sample_acmtog.tex sample-acmtog.tex
	@echo "Done."
