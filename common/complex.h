
/* $Id$ */

#ifndef COMPLEX_H
#define COMPLEX_H

#include <iostream>
using namespace std;
#include <cmath>
#include <cstdlib>
#include <cstdio>

#ifndef Pi
#define Pi M_PI //3.14159265358979323846264338328
#endif

class Complex
{
 private:
  double re, im;
  
 public:
  
  Complex(double r = 0, double i = 0) : re(r), im(i) { }
  Complex(const Complex &c) : re(c.re), im(c.im) { }
  
  operator double *()
  { return (double *) this; }
  
  operator const double *() const { return (const double *) this; }
  
  double real() const { return re; }
  double imag() const { return im; }
  
  void set_from_polar(double r=0.0, double theta = 0.0)
  { 
    re = r*cos(theta);
    im = r*sin(theta);
  }
  
  friend double real(const Complex &c) { return c.re; }
  friend double imag(const Complex &c) { return c.im; }
  friend Complex conj(const Complex &c) { return Complex(c.re, -c.im); }

  friend double abs2(const Complex &c)
  { return c.re*c.re + c.im*c.im; }
  
  friend double abs(const Complex &c)
  { return sqrt(abs2(c)); }
  
  friend double norm(const Complex &c)
    { return abs(c); }

  double norm() const
  { return abs(*this); }
  
  double magnitude() const
  { return abs2(*this); }
  
  friend double arg(const Complex &c)
  {
    double angle = asin(c.im/abs(c));
    if(c.re < 0 && c.im > 0)
      return Pi - angle;
    if(c.re < 0 && c.im < 0)
      return -(Pi+angle);
    return angle;
  }
  
  Complex & operator =(const Complex &c)
  { 
    if(this != &c) {
      re = c.re;
      im = c.im;
    }
    return *this;
  }
  
  friend int operator ==(const Complex &c1, const Complex &c2)
  { return (c1.re == c2.re && c1.im == c2.im); }
  
  friend int operator !=(const Complex &c1, const Complex &c2)
  { return (c1.re != c2.re || c1.im != c2.im); }
  
  Complex operator -() const
  { return Complex(-re, -im); }
  
  Complex & operator +=(const Complex &c)
  { re += c.re; im += c.im; return *this; }
  Complex & operator -=(const  Complex &c)
  { re -= c.re; im -= c.im; return *this; }
  Complex & operator +=(double r)
  { re += r; return *this; }
  Complex & operator -=(double r)
  { re -= r; return *this; }
  Complex & operator *=(double r)
  { re *= r; im *= r; return *this; }
  Complex & operator /=(double r)
  { return *this *= 1.0/r; }
  Complex & operator *=(const Complex &c)
  { return *this = *this * c; }
  Complex & operator /=(const Complex &c)
  { return *this = *this / c; }
  
  friend Complex operator +(double r, const Complex &c)
  { return Complex(r+c.re, c.im); }
  friend Complex operator +(const Complex &c, double r)
  { return r + c; }
  friend Complex operator +(const Complex &c1, const Complex &c2)
  { return Complex(c1.re+c2.re, c1.im+c2.im); }
  
  friend Complex operator -(double r, const Complex &c)
  { return Complex(r-c.re, -c.im); }
  friend Complex operator -(const Complex &c, double r)
  { return Complex(c.re-r, c.im); }
  friend Complex operator -(const Complex &c1, const Complex &c2)
  { return Complex(c1.re-c2.re, c1.im-c2.im); }
  
  friend Complex operator *(const Complex &c, double r)
  { return Complex(c.re*r, c.im*r); }
  friend Complex operator *(double r, const Complex &c) 
  { return c*r; }
  friend Complex operator *(const Complex &c1, const Complex &c2)
  { return Complex(c1.re*c2.re - c1.im*c2.im, c1.re*c2.im + c1.im*c2.re); }
  
  friend Complex operator /(const Complex &c, double r)
  { return Complex(c.re/r, c.im/r); }
  friend Complex operator /(double r, const Complex &c)
  { return r/abs2(c) * conj(c); }
  friend Complex operator /(const Complex &c1, const Complex &c2)
  { return c1/abs2(c2) * conj(c2); }
  
  friend Complex exp(const Complex &c)
  { return exp(c.re) * Complex(cos(c.im), sin(c.im)); }
  friend Complex sinh(const Complex &c)
  { return (exp(c) - exp(-c)) / 2.0; }
  friend Complex cosh(const Complex &c)
  { return (exp(c) + exp(-c)) / 2.0; }
  friend Complex tanh(const Complex &c)
  { return sinh(c)/cosh(c); }
  friend Complex sin(const Complex &c)
  { return sinh(Complex(0,1.0) * c) / Complex(0,1.0); }
  friend Complex cos(const Complex &c)
  { return cosh(Complex(0,1.0) * c); }
  friend Complex tan(const Complex &c)
  { return sin(c)/cos(c); }
  
  friend Complex pow(const Complex &c, double n)
  {
    double theta = n*arg(c);
    double r = pow(abs2(c), n/2.0);
    
    return Complex(r*cos(theta), r*sin(theta));
  }
  
  friend Complex sqrt(const Complex &c)
  {
    return pow(c, 0.5);
  }
  
  friend Complex pow(double r, const Complex &c)
  { return exp(c*log(r)); }
  
  /********************************************************
   **
   ** Maybe there are something wrong with 'log' operator
   ** when dealing with log(-1) and the similar special
   ** points
   **
   ** *****************************************************/
  
  friend Complex log(const Complex &c)
  {
    double theta = arg(c);
    double rho = abs(c);
    return log(rho) + Complex(0,1.0)*theta;
  }
  
  friend Complex pow(const Complex &c1, const Complex &c2)
  { return exp(c2*log(c1)); }
  
  friend ostream & operator <<(ostream &s, const Complex &c)
  { return s << "(" << c.re << ", " << c.im << ")"; }

  void printf() const
  { ::printf("(%.8f, %.8f)\n", re, im); }
};

#endif /* COMPLEX_H */  
