
/* $Id$ */

#ifndef MATLAB_STRUCTURES_H
#define MATLAB_STRUCTURES_H

#include <iostream>
using namespace std;
#include <mex.h>
#include "rmat.h"

class RadialCoordinate
{
public:
  
  friend class TimeEvolution;
  
  const int &n; // out
  RVec r;
  const double &dr; // out
  const double &mass; // out
  
  RVec psq2m;
  RVec one2mr2;
  
  RadialCoordinate(const mxArray *mx);
  
private:
  
  const mxArray *mx;

  // to prevent assigment and copy operation
  RadialCoordinate(const RadialCoordinate &);
  RadialCoordinate & operator =(const RadialCoordinate &);
  
  /* IO */
  friend ostream & operator <<(ostream &s, const RadialCoordinate &c);
  void write_fields(ostream &s) const;
};

class AngleCoordinate
{
public:
  
  friend class TimeEvolution;
  
  const int &n; // out
  const int &m; // out
  RVec x;
  RVec w;
  RMat legendre; 

  AngleCoordinate(const mxArray *mx);

private:
  
  const mxArray *mx;

  // to prevent assigment and copy operation
  AngleCoordinate(const AngleCoordinate &);
  AngleCoordinate & operator =(const AngleCoordinate &);

  /* IO */
  friend ostream & operator <<(ostream &s, const AngleCoordinate &c);
  void write_fields(ostream &s) const;
};

class EvolutionTime
{
public:

  friend class TimeEvolution;

  const int &total_steps; // out
  const double &time_step; // out
  int &steps; // out

  EvolutionTime(const mxArray *mx);

private:
  
  const mxArray *mx;

  EvolutionTime(const EvolutionTime &);
  EvolutionTime & operator =(const EvolutionTime &);
  
  /* IO */
  friend ostream & operator <<(ostream &s, const EvolutionTime &c);
  void write_fields(ostream &s) const;
};

class Options
{
public:
  
  friend class TimeEvolution;

  char *wave_to_matlab; // out
  char *test_name; // out
  const int &steps_to_copy_psi_from_device_to_host; // out

  Options(const mxArray *mx);
  ~Options();

private:

  const mxArray *mx;

  Options(const Options &);
  Options & operator =(const Options &);

  friend ostream & operator <<(ostream &s, const Options &c);
  void write_fields(ostream &s) const;
};

class DumpFunction
{
public:
  
  friend class TimeEvolution;

  DumpFunction(const mxArray *mx);
  ~DumpFunction();

  double *dump; 

private:
  
  const mxArray *mx;
};

class CummulativeReactionProbabilities
{
public:

  friend class TimeEvolution;

  RVec energies; // out
  RVec eta_sq; // out
  RVec CRP; // out

  const int &n_dividing_surface; // out
  const int &n_gradient_points; // out
  const int &n_energies; // out
  const int &calculate_CRP; // out
  
  CummulativeReactionProbabilities(const mxArray *mx);

private:
  const mxArray *mx;

  CummulativeReactionProbabilities(const CummulativeReactionProbabilities &);
  CummulativeReactionProbabilities & operator =(const CummulativeReactionProbabilities &);

  friend ostream & operator <<(ostream &s, const CummulativeReactionProbabilities &c);
  void write_fields(ostream &s) const;
  
};

#endif /* MATLAB_STRUCTURES_H */
