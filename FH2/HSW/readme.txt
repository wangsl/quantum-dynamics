hswpot.f   : this subroutine calculates the value of the HSW-potential in
	     atomic units, given one point in distance coordinates:
                r(F-H1), r(H1-H2), r(F-H2)     (in atomic units)
	     This routine initializes itself on the first call.
             For further explanations, please consult the comments in the
	     program text.
sospline.f : spline interpolation of the spin-orbit correction data. This
	     routine is called internally by "hswpot.f". You do not have
             to call this routine yourself, but you have to link it together 
	     with hswpot to your application programs.
so.input   : the spin-orbit correction data points.
so.param   : parameters for the spline interpolation.
three.param: three-body parameters for the SW1 surface.
two.param  : two-body parameters for the SW1 surface.
verify.f   : a trivial test program (main program) to verify your installation.

After downloading, you should test your installation by compiling the three
*.f-files into one executable. This executable attempts to calculate the
potential at two points. The calculated values should agree to almost-all
digits with our values that are also returned by the program.
After this test, you can discard "verify.f" and call "hswpot" from your
own programs.

For questions and comments, please contact one of the following addresses:

Prof.Dr. H.-J.Werner:  werner@theochem.uni-stuttgart.de
        Dr. B.Hartke:  hartke@theochem.uni-stuttgart.de
