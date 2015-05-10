#!/bin/bash

# $Id$

git add *.[FChm] *.pl *.sh Makefile POT4.dat  POT4P.dat  POT5.dat  POT5L.dat  POT5P.dat  POTasy.dat  so.input  so.param

#svnId *.[FChm] *.pl *.sh *.inc Makefile

exit

git commit -m "comments"

git push origin master



