      
      
      double precision function pot(rhf1,r,rhf2, ReadDataOnly)
c given one point in distance coordinates (bohr), this routine returns the
c value of the HSW potential (hartree), or, optionally (see comments below)
c the value of the SW1 potential (hartree).
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c This is a spin-orbit corrected version of the SW1 potential energy surface, c 
c by B. Hartke, K. Stark and H.-J. Werner.                                    c
c The SW1 potential without S.O.-correction has been published in:            c
c   K. Stark and H.-J. Werner, J.Chem.Phys. 104 (1996) 6515.                  c
c For work using both surfaces, see:                                          c
c   B. Hartke and H.-J. Werner, Chem.Phys.Lett. 280 (1997) 430.               c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit double precision(a-h,o-z)
      parameter (np=40)
      dimension x(np),y(np),z(np),en(np,np,np)
      dimension rr(3),fin(4)
      data amuau,evau,aukcal/1822.88734d0,27.211608d0,627.5095d0/
      data ifirst/-1/
      save ifirst,pi,raddeg,nx,ny,nz,x,y,z,en,esocut,rsolim,rrsolm

      Logical ReadDataOnly
      Include 'hswdata.inc'
c
c on first call of this subroutine, initialize the SW1 potential routine
c and the spline interpolation for the S.O. correction
c
      if (ifirst.eq.-1) then
         call inifit
         call prepar(nx,ny,nz,x,y,z,en)
         open(unit=83,file=DataDir(1:DataDirLength)//'/so.param',
     $        status='old')
         read(83,*)esocut
         read(83,*)rsolim
         read(83,*)rrsolm
         close(83)
         esocut=esocut/aukcal
         pi=acos(-1.0d0)
         raddeg=180.d0/pi
         ifirst=0
      end if

      If(ReadDataOnly) Return

c call sw1 potential routine
      call sw1(rhf1,r,rhf2,v)
c
c convert point to F + H_2 Jacobi coordinates, for convenience
      rsmall=r
      rbig=sqrt(0.5d0*(rhf1*rhf1+rhf2*rhf2-0.5d0*r*r))
      thn=(rbig*rbig+0.25d0*r*r-rhf1*rhf1)/rbig/r
c depending on the computer, the above may result in abs(thn) being slightly
c larger than 1.0, due to rounding errors; hence correct for that:
      if (abs(thn).ge.1.d0) then
         thn=sign(1.d0,thn)
      end if
      thn=acos(thn)
c check if point is in F+H2 valley; criteria:
c (1) energy is larger than some cutoff value esocut (this is the most
c convenient way to exclude the HF+H valley)
c (2) the H-H distance ( = r ) is smaller than some limit rsolim
c (3) the F-H2 distance ( = R ) is larger than some limit rrsolm
      if (v.gt.esocut.and.r.lt.rsolim.and.rbig.gt.rrsolm) then
c point is in F+H2 valley, so add spline-interpolated S.O.-correction
c (note: the S.O. correction is given in F+H2 Jacobi coordinates, with
c the angle not in radians but in degree!).
         rr(1)=rbig
         rr(2)=r
         rr(3)=thn*raddeg
c First take care of points outside the actual x,y,z-grid of S.O.-data:
c if point is outside interpolation grid, "project it back" onto the
c border of the interpolation grid (this corresponds to extending the
c interpolation grid with constant function values in all directions):
         if (rr(1).lt.x(1)) rr(1)=x(1)
         if (rr(1).gt.x(nx)) rr(1)=x(nx)
         if (rr(2).lt.y(1)) rr(2)=y(1)
         if (rr(2).gt.y(ny)) rr(2)=y(ny)
c since z is actually gamma (or theta) here, we complement the given
c z- (or gamma-) range of 0--90 to 0--180 (note: this is already the full
c coordinate range, no extension necessary here!) by setting z=180-gamma for
c gamma-values > 90.
         if (rr(3).gt.90.d0) rr(3)=180.d0-rr(3)
c do the spline interpolation to this point:
         call naturl(rr,0,fin)
         socorr=fin(1)
c if you want derivatives, just call the spline routine again:
c         call naturl(rr,3,fin)
c fin(2),fin(3),fin(4) now contain the x,y,z-derivatives of the S.O.correction
      else
c point is outside F+H2 valley, hence spin-orbit correction should be zero:
         socorr=0.d0
      end if

c add in the asymptotic value of the spin-orbit correction, in order to
c shift the bottom of the asymptotic F+H2 valley to zero again, just
c to make life easier...
      pot=v+socorr+0.0006
c in order to get the SW1 potential, just omit the S.O. correction and shift
c      pot=v
c in that case, you can also comment out all the above parts relating to
c computation of the "socorr" value (spline interpolation, etc.)
      return
      end
c------------------------------------------------------------------------
      subroutine inifit
      implicit real*8(a-h,o-z)
      parameter(nmx=200)
      common/cparm/ a(nmx),p(nmx),ma,np
      common/cint/ ix(nmx),iy(nmx),iz(nmx),mmax
      include 'hswdata.inc'
	  open(1,file=DataDir(1:DataDirLength)//'/three.param',status='old')
	  i=1
      rewind 1
10    read(1,*,end=100) nparm,ix(i),iy(i),iz(i),a(i)
      i=i+1
	  goto 10
100   ma=i-1
      close(1)
      m=ix(ma-6)
      mmax=m+1
	  open(1,file=DataDir(1:DataDirLength)//'/two.param',status='old')
	  i=1
      rewind 1
11     read(1,*,end=111)p(i)
      i=i+1
	  goto 11
111   np=i-1
      close(1)
      return
      end

      subroutine sw1(x,y,z,v)
      implicit real*8(a-h,o-z)
      parameter(nmx=200)
      common/cparm/ a(nmx),p(nmx),ma,np
      common/cint/ ix(nmx),iy(nmx),iz(nmx),mmax
      dimension xex(0:mmax),xey(0:mmax),xez(0:mmax)
c
c.... calculates potential of the Stark-Werner surface for F+H2
c.... initializing the non-linear parameters
c....  x,y,z are the internal coordinates of F-H1-H2 :
c....  x = r(F-H1) y= r(H1-H2) z= r(F-H2)
c
      b1 = a(ma-5)
      b2 = a(ma-4)
      b3 = a(ma-3)
      x0 = a(ma-2)
      y0 = a(ma-1)
      z0 = a(ma)
      fit = 0.0d0
      xexpon = b1*(x-x0)
      yexpon = b2*(y-y0)
      zexpon = b3*(z-z0)
      exponx=dexp(-xexpon)
      expony=dexp(-yexpon)
      exponz=dexp(-zexpon)
      fex = x*exponx
      fey = y*expony
      fez = z*exponz
      xex(0)=1
      xey(0)=1
      xez(0)=1
      xex(1)=fex
      xey(1)=fey
      xez(1)=fez
      do m=2,mmax-1
	 xex(m)=xex(m-1)*fex
	 xey(m)=xey(m-1)*fey
	 xez(m)=xez(m-1)*fez
       enddo


c
c.... Aguado-Paniagua-Fit for the threebody-terms
c.... Method seems to be a bit confusing
c.... But this type of programming was chosen to avoid
c.... terms like x**b(i)*y**c(j)*z**d(k), which
c.... cost much more CPU-time !
c
      do 1010 i=1,ma-6
       fit=fit+xex(ix(i))*xey(iy(i))*xez(iz(i))*a(i)
1010  continue
c
c.... Two-Body-Potential : Extended-Rydberg-Functional
c
      xr = x-p(3)
      yr = y-p(9)
      zr = z-p(3)
      xr2=xr*xr
      xr3=xr2*xr
      yr2=yr*yr
      yr3=yr2*yr
      zr2=zr*zr
      zr3=zr2*zr
      fx = dexp(-p(2)*xr)
      fy = dexp(-p(8)*yr)
      fz = dexp(-p(2)*zr)
      ux = -p(1)*(1.0d0+p(2)*xr+p(4)*xr2+p(5)*xr3)
      uy = -p(7)*(1.0d0+p(8)*yr+p(10)*yr2+p(11)*yr3)
      uz = -p(1)*(1.0d0+p(2)*zr+p(4)*zr2+p(5)*zr3)
      xval=ux*fx+p(6)
      yval=uy*fy+p(12)
      zval=uz*fz+p(6)
c
c... Resulting Potential in atomic untis
c
      v=fit+xval+yval+zval
      return
      end
