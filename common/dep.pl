#!/bin/env perl

# $Id$

require 5.002;
use Getopt::Std;

$a = '';

while(<>) {
  s/\/usr\S*\.h//g; 
  s/\w*\.[Cc]//g;
  if (/\\$/) {
    s/\s*\\$//;
    chop;
  }
  s/ +/ /g;
  s/(\S+\.o)/$a$1/g;
  print;
} 

#$allfiles = join " ", grep !/rcsid.C/, glob( "*.[hcCfF]" );
#print "rcsid.C: $allfiles\n";
#print "\t perl rcsid.pl > rcsid.C\n";
