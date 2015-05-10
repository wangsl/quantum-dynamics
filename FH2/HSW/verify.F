      program verify
c to check the correct implementation of the HSW potential for the FH2 neutral
      implicit double precision(a-h,o-z)
      write(6,*)'first test point:'
      ftrue=3.412633992079376E-02
      x=2.d0
      y=2.d0
      z=2.d0
      f=pot(x,y,z)
      write(6,*)'your calculated value: ',f
      write(6,*)'true value:            ',ftrue
      write(6,*)
      write(6,*)'second test point:'
      ftrue=1.839160872060410E-03
      x=4.d0
      y=1.5d0
      z=4.5d0
      f=pot(x,y,z)
      write(6,*)'your calculated value: ',f
      write(6,*)'true value:            ',ftrue
      end
