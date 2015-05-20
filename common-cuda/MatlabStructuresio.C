
/* created at: 2015-03-24 21:22:04 */

#include <iostream>
using namespace std;
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "indent.h"
#include "MatlabStructures.h"
#include "die.h"

ostream & operator <<(ostream &s, const RadialCoordinate &c)
{
  s << " {\n";
  IndentPush();
  c.write_fields(s);
  IndentPop();
  return s << Indent() << " }";
}

void RadialCoordinate::write_fields(ostream &s) const
{
  s << Indent() << "n " << n << "\n";
  s << Indent() << "dr " << dr << "\n";
  s << Indent() << "mass " << mass << "\n";
}

ostream & operator <<(ostream &s, const AngleCoordinate &c)
{
  s << " {\n";
  IndentPush();
  c.write_fields(s);
  IndentPop();
  return s << Indent() << " }";
}

void AngleCoordinate::write_fields(ostream &s) const
{
  s << Indent() << "n " << n << "\n";
  s << Indent() << "m " << m << "\n";
}

ostream & operator <<(ostream &s, const EvolutionTime &c)
{
  s << " {\n";
  IndentPush();
  c.write_fields(s);
  IndentPop();
  return s << Indent() << " }";
}

void EvolutionTime::write_fields(ostream &s) const
{
  s << Indent() << "total_steps " << total_steps << "\n";
  s << Indent() << "time_step " << time_step << "\n";
  s << Indent() << "steps " << steps << "\n";
}

ostream & operator <<(ostream &s, const Options &c)
{
  s << " {\n";
  IndentPush();
  c.write_fields(s);
  IndentPop();
  return s << Indent() << " }";
}

void Options::write_fields(ostream &s) const
{
  if (wave_to_matlab)
    s << Indent() << "wave_to_matlab " << wave_to_matlab << "\n";
  if (test_name)
    s << Indent() << "test_name " << test_name << "\n";
}

ostream & operator <<(ostream &s, const CummulativeReactionProbabilities &c)
{
  s << " {\n";
  IndentPush();
  c.write_fields(s);
  IndentPop();
  return s << Indent() << " }";
}

void CummulativeReactionProbabilities::write_fields(ostream &s) const
{
  s << Indent() << "n_dividing_surface " << n_dividing_surface << "\n";
  s << Indent() << "n_gradient_points " << n_gradient_points << "\n";
  s << Indent() << "n_energies " << n_energies << "\n";
  s << Indent() << "calculate_CRP " << calculate_CRP << "\n";
  s << Indent() << "energies " << energies << "\n";
  s << Indent() << "eta_sq " << eta_sq << "\n";
  s << Indent() << "CRP " << CRP << "\n";
}

