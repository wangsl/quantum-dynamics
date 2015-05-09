c
c ifort -132 -save fh2FXZ
c a test program for fh2FXZ
c

#if 0 
       implicit real*8 (a-h,o-z)
       dimension rin(3)

       rr1=0.5d0; rr2=4.d0; N=100
       DR=(rr2-rr1)/N

c asympotoic PES for H2
        do i=1,N
          rin(1)=rr1+(i-1)*DR
          rin(2)=10.d0
          rin(3)=rin(1)+rin(2)
          call fh2FXZ(rin,Vout)
          rin(1)=rr1+(i-1)*DR
          write(1,*) rin(1),Vout*27.2116d0
        end do

c asympotoic PES for HF
        do i=1,N
          rin(1)=10.D0
          rin(2)=rr1+(i-1)*DR
          rin(3)=rin(1)+rin(2)
          call fh2FXZ(rin,Vout)
          rin(2)=rr1+(i-1)*DR
          write(2,*) rin(2),Vout*27.2116d0
        end do

        stop
        end
#endif
c#####################################################################
	subroutine fh2FXZ(rin00,Vout, ReadDataOnly)
	implicit real*8 (a-h,o-z)
	parameter (MN=70,MNA=41)

        Logical ReadDataOnly
c
c rin00 are three bond length: rHH', rHF, and rH'F in Bohr.
c Vout is in Hartree with zero defined as F+H$_2$($r=r_e$)
c
c (N2P4,r1P4,r2P4,V4P,V4Pd) data for AVQZ perpendicular line
c data for AVQZ/AV5Z Perpendicular potential 
	dimension r1P4(MN),r2P4(MN),V4P(MN,MN),V4Pd(MN,MN)
	dimension r1P5(MN),r2P5(MN),V5P(MN,MN),V5Pd(MN,MN)

c data for AVQZ/AV5Z Linear potential 
	dimension r1L4(MN),r2L4(MN),V4L(MN,MN),V4Ld(MN,MN)
	dimension r1L5(MN),r2L5(MN),V5L(MN,MN),V5Ld(MN,MN)

c data for AV4Z/AV5Z 3D potential 
	dimension r1(MN),r2(MN),theta(MNA,MN,MN),Nt(MN,MN),
     $  V4(MNA,MN,MN),V4d(MNA,MN,MN)

	dimension s1(MN),s2(MN),theta5(MNA,MN,MN),Nt5(MN,MN),
     $  V5(MNA,MN,MN),V5d(MNA,MN,MN)     ! AV5Z DATA

        dimension V1d(MN),V1d5(MN),WK(MN),WKp(MN)
        dimension r2WK(MN),rV2(MN),V2d(MN,MN),V2p(MN,MN)
	dimension rin0(3),rin(3),r1inf(MN),r2inf(MN),rin00(3)
 	dimension H2(MN),HF(MN),H2d(MN),HFd(MN),r0(MN)

        data init/0/
c        save init,r1,r2,theta,Nt,V4,V4p
        save init
	
	rin0=rin00
	pi=acos(-1.d0);  a0=0.529177d0;  eV=27.2116d0
	scale=1.01d0;  r1cut=4.d0;  r2cut=4.2d0; 
	V40=100.82955298935d0;  V50=100.838761085897d0
	Vout=8.d0/27.2116d0; Vmax=Vout
	
c---------------------------------------------------------
c read in potential data
c---------------------------------------------------------
        if(init.eq.0) then
  	  init=1

	  OPEN(580,FILE="POTasy.dat",status='OLD')
	  N=0
	  DO
	    read(580,*,end=18) r0(N+1),H2(N+1),x,HF(N+1)
	    N=N+1
	  END DO
18	  Nasy=N
	  dvdr1=(H2(2)-H2(1))/(r0(2)-r0(1))
	  dvdr2=(HF(2)-HF(1))/(r0(2)-r0(1))
	  call spline(Nasy,r0,H2,dvdr1,0.d0,H2d)
	  call spline(Nasy,r0,HF,dvdr2,0.d0,HFd)
	  CLOSE(580)

c (N2P4,r1P4,r2P4,V4P,V4Pd) data for AVQZ perpendicular line
	  OPEN(600,FILE="POT4P.dat",status='OLD')
	  CALL POTread2d(600,MN,N1,N2P4,r1P4,r2P4,V4P,V4Pd,scale,V40,0.D0,
     $    N1p,N2P4,r1P4,r2P4,V4P)
	  CLOSE(600)

c	  OPEN(700,FILE="POT4L.dat",status='OLD')
c	  CALL POTread2d(700,MN,N1p,N2L4,r1L4,r2L4,V4L,V4Ld,scale,V40,0.D0,
c     $    N1p,N2L4,r1L4,r2L4,V4L)
c	  IF(N1.NE.N1p) STOP 'N1 <> N1p'
c	  CLOSE(700)
c	  print*,'finish reading 2D AVQZ data'

c ===  V5P -V4P ===
	  OPEN(650,FILE="POT5P.dat",status='OLD')
	  CALL POTread2d(650,MN,M1,N2P5,r1P5,r2P5,V5P,V5Pd,scale,V50,0.D0,
     $    N1,N2P4,r1P4,r2P4,V4P)
	  CLOSE(650)

	  OPEN(700,FILE="POT5L.dat",status='OLD')
	  CALL POTread2d(700,MN,M1p,N2L5,r1L5,r2L5,V5L,V5Ld,scale,V50,0.D0,
     $    N1p,N2L4,r1L4,r2L4,V4L)
	  IF(M1.NE.M1p) STOP 'M1 <> N1p'
	  CLOSE(700)
	  print*,'finish reading 2D AV5Z data'

c read in 3D AV4Z/AV5Z  potential
	  OPEN(500,FILE="POT4.dat",status='OLD')
	  call POTread3d(500,MN,N1,N2,r1,r2,MNA,Nt,theta,V4,V4d,
     $    scale,V40,N1cut,N2cut,N2P4,r2P4,V4P,V4Pd)
	  CLOSE(500)

	  OPEN(550,FILE="POT5.dat",status='OLD')
	  call POTread5Z(550,MN,M1,M2,s1,s2,MNA,Nt5,theta5,V5,V5d,
     $    scale,V50,N1cut5,N2cut5,N1,N2,r1,r2,Nt,theta,V4,V4d,
     $    N2P5,r2P5,V5P,V5Pd,N2L5,r2L5,V5L)
	  CLOSE(550)
	  print*,'finish reading 3D AVQZ/AV5Z data'

	END IF  ! end of init

        If(ReadDataOnly) Return

c---------------------------------------------------------
c set limits and
c exchange rin0(2) and rin0(3) if rin0(2) > rin0(3)
c---------------------------------------------------------
        r1min=r1(1); r2min=r2(1); r1max=r1(N1); r2max=r2(N2)
        r1min5=s1(1); r2min5=s2(1)

        if(rin0(3).lt.rin0(2)) then
          tmp=rin0(3); rin0(3)=rin0(2); rin0(2)=tmp
        end if


        if(rin0(1).le.r1min) return
        if(rin0(2).le.r2min) return

	vsp=pot_FHH_HSW(min(rin0(2),10.d0),min(rin0(1),11.d0),min(rin0(3),10.d0))
c        rin0(1)=min(rin0(1),11.d0)
c        rin0(2)=min(rin0(2),10.d0)

	if(rin0(1).gt.r1max+100.d0) then
          call splint(Nasy,r0,HF,HFd,rin0(2),Vout)
	  Vout=Vout+vsp
	  return
	end if

	if(rin0(2).gt.r2max+100.d0) then
          call splint(Nasy,r0,H2,H2d,rin0(1),Vout)
	  Vout=Vout+vsp
	  return
	end if

        rin0(1)=min(rin0(1),r1max)
        rin0(2)=min(rin0(2),r2max)


        if(rin0(1).GT.r1cut .and. rin0(2).GT.r2cut) then
	  VoutQ=4.42d0/eV
          Vout=4.42d0/eV; return
        end if

        theta1=(rin0(1)*rin0(1)+rin0(2)*rin0(2)-rin0(3)*rin0(3))
     $         /(2.d0*rin0(1)*rin0(2))
        theta1=min(1.d0,theta1); theta1=max(-1.d0,theta1)
        theta1=acos(theta1)

c        theta0=acos(min(rin0(1)/(2.d0*rin0(2)),1.d0-1.d-20))
c -------------------  Start spline --------------------------
c spline potential on every r1(i) for fixed rin0(2) and theta1,
c so should consider the symmetry relationship between
c r1(i) and rin0(2) 
c -----------------------------------------------------------
	DO i=4,N1cut5
	  CALL INT2D(1,V1d5(i),s1(i),rin0(2),theta1,r2min5,Vout5,M2,s2,
     $         N2P5,r2P5,MNA,V5P(1,i),V5Pd(1,i),r2WK,Nt5(1,i),theta5(1,1,i),
     $         V5(1,1,i),V5d(1,1,i),WK,WKp)
	END DO   ! end of N1cut loop

	DO i=1,N1cut
	  CALL INT2D(1,V1d(i),r1(i),rin0(2),theta1,r2min,Vout,N2,r2,
     $         N2P4,r2P4,MNA,V4P(1,i),V4Pd(1,i),r2WK,Nt(1,i),theta(1,1,i),
     $         V4(1,1,i),V4d(1,1,i),WK,WKp)
	END DO

	IF(rin0(2).LE.r2cut) THEN
          DO i=N1cut+1,N1
	     CALL INT2D(2,V1d(i),r1(i),rin0(2),theta1,r2min,Vout,N2cut,r2,
     $         N2P4,r2P4,MNA,V4P(1,i),V4Pd(1,i),r2WK,Nt(1,i),theta(1,1,i),
     $         V4(1,1,i),V4d(1,1,i),WK,WKp)
	  END DO  ! end do of N1cut loop

	  DO i=N1cut5+1,M1
	     CALL INT2D(2,V1d5(i),s1(i),rin0(2),theta1,r2min5,Vout5,N2cut5,s2,
     $         N2P5,r2P5,MNA,V5P(1,i),V5Pd(1,i),r2WK,Nt5(1,i),theta5(1,1,i),
     $         V5(1,1,i),V5d(1,1,i),WK,WKp)
	  END DO

	  dvdr=(V1d(2)-V1d(1))/(r1(2)-r1(1))
	  call spline(N1,r1,V1d,dvdr,0.d0,WKp)
          call splint(N1,r1,V1d,WKp,rin0(1),Vout)

	  dvdr=(V1d5(2)-V1d5(1))/(s1(2)-s1(1))
	  call spline(M1,s1,V1d5,dvdr,0.d0,WKp)
          call splint(M1,s1,V1d5,WKp,rin0(1),Vout5)
	  VoutQ=Vout
	  Vout=Vout+Vout5

	ELSE  !    rin0(2) > r2cut
	  dvdr=(V1d(2)-V1d(1))/(r1(2)-r1(1))

c add 3 more points from symmetric line
	  DO k=N1cut+1,N1cut+3
	    Dist=rin0(2)*rin0(2)-r1(k)*r1(k)/4.d0
	    IF(Dist.LE.0.D0) Stop 'Dist < 0'
	    Dist=sqrt(Dist)
	    CALL splint(N2P4,r2P4,V4P(1,k),V4Pd(1,k),Dist,V1d(k))
	  end do
	  call spline(N1cut+3,r1,V1d,dvdr,0.d0,WKp)
          call splint(N1cut+3,r1,V1d,WKp,rin0(1),Vout)
	  VoutQ=Vout

          dvdr=(V1d5(2)-V1d5(1))/(s1(2)-s1(1))
          DO k=N1cut5+1,N1cut5+3
            Dist=rin0(2)*rin0(2)-s1(k)*s1(k)/4.d0
            IF(Dist.LE.0.D0) Stop 'Dist < 0'
            Dist=sqrt(Dist)
            CALL splint(N2P5,r2P5,V5P(1,k),V5Pd(1,k),Dist,V1d5(k))
          end do
          call spline(N1cut5+3,s1,V1d5,dvdr,0.d0,WKp)
          call splint(N1cut5+3,s1,V1d5,WKp,rin0(1),Vout5)
          Vout=Vout+Vout5

	END IF
	Vout=min(Vmax,Vout)
	Vout=Vout+vsp
	  
	return
	end
c=====================================================================
	SUBROUTINE INT2D(ID,V1d,r10,r20,theta1,r2min,Vout,N2,r2,
     $  Nv2,rV2,MNA,V2d,V2p,r2WK,Nt,theta,V4,V4p,WK,WKp)
	IMPLICIT REAL*8 (A-H,O-Z)
	DIMENSION r2(N2),rV2(Nv2),V2d(1),V2p(1),r2WK(N2),
     $  Nt(1),theta(MNA,1),V4(MNA,1),V4p(MNA,1),WK(1),WKp(1)

	  rr1=r10; rr2=r20; thetaS=theta1; pi=acos(-1.d0)

c Check if exchange rin0(2) with another bond for this r1
          rr3=sqrt(rr1*rr1+rr2*rr2-2.d0*rr1*rr2*cos(thetaS))

	  IF (rr3.LT.rr2) THEN  ! exchange
            tmp=rr3; rr3=rr2; rr2=tmp
	    thetaS=(rr1*rr1+rr2*rr2-rr3*rr3)/(2.d0*rr1*rr2)
	    thetaS=min(1.d0,thetaS); thetaS=max(-1.d0,thetaS)
	    thetaS=acos(thetaS)
	  END IF

	  r2Spline=rr2; thetaSpline=thetaS
	  IF (r2Spline.LE.r2min) THEN
	     V1d=Vout; RETURN
	  END IF
c	  print*,r2Spline,thetaSpline

c for the case thetaSpline > pi/2, use all N2 points 
	  N2Spline=N2; r2WK=r2; Epsymm=0.d0; N2Splinef=N2Spline
	  
	  Dist=r10/2.d0*tan(thetaSpline)
c	  print*,Dist,rV2(Nv2)
c	  IF (thetaSpline .LT. pi/2.d0) THEN
	  IF (Dist.LE.rV2(Nv2) .AND. thetaSpline .LT. pi/2.d0) THEN
c find the cross point to symmetry line and the number of points 
	    r2symm=r10/2.d0/cos(thetaSpline)   ! distance to symmetric line
	    DO k=1,N2
	     IF (r2(k).GT.r2symm) GO TO 3
	    END DO
3	    N2Spline=k-1

c calculate energy and first derivate with respect to r2 on the cross point
	    Dist=r10/2.d0*tan(thetaSpline)
	    CALL SPLINTp(Nv2,rV2,V2d,V2p,Dist,Esymm,Epsymm)
	    Epsymm=Epsymm*sin(thetaSpline)

c add the point on symmetric line if it differs with the last grid point of r2.
	    IF(abs(r2WK(N2Spline)-r2symm).GT.1.D-5) THEN
	      N2Splinef=N2Spline+1
              r2WK(N2Splinef)=r2symm; WK(N2Splinef)=Esymm
	    END IF
	  END IF

	  DO j=1,N2Spline
            call splint(Nt(j),theta(1,j),V4(1,j),V4p(1,j),thetaSpline,WK(j))
	  END DO

c add 3 more points for asymptotic energy
	  IF(ID.EQ.2) THEN
	    Dr=0.5D0
	    DO k=1,3
	      r2WK(N2Splinef+k)=r2WK(N2Splinef)+Dr*k
	      Dist=r2WK(N2Splinef+k)*r2WK(N2Splinef+k)-r10*r10/4.d0
	      Dist=sqrt(max(Dist,0.d0))
c	      print*,k,Dist,r10
	      CALL SPLINT(Nv2,rV2,V2d,V2p,Dist,Esymm)
	      WK(N2Splinef+k)=Esymm
	    END DO
	    Epsymm=0.D0
	    N2Splinef=N2Splinef+3
	  END IF

	  IF(N2Splinef.GE.2) THEN
	    dvdr=(WK(2)-WK(1))/(r2WK(2)-r2WK(1))
	    call spline(N2Splinef,r2WK,WK,dvdr,Epsymm,WKp)
	    call splint(N2Splinef,r2WK,WK,WKp,r2Spline,V1d)
	  ELSE
c	    V1d=Esymm+Epsymm*(rin0(2)-r2V)   ! use linear extropolation
	    V1d=Esymm                     
	  END IF

	RETURN
	END
c#########################################################################
	subroutine POTread5Z(IDpot,MN,N1,N2,r1,r2,MNA,Nt,theta,V,Vp,
     $  scale,V00,N1cut,N2cut,M1,M2,s1,s2,Nt4,theta4,V4,V4d,
     $  N2P5,r2P5,V5P,V5Pd,N2L5,r2L5,V5L)
	implicit real*8 (a-h,o-z)
	dimension V(MNA,MN,MN),Vp(MNA,MN,MN),r1(1),r2(1),theta(MNA,MN,MN),
     $  V1d(MN),X0(MN),WK(MN),F(2),Nt(MN,1),Ed(2),
     $  s1(MN),s2(MN),Nt4(MN,MN),theta4(MNA,MN,1),V4(MNA,MN,1),V4d(MNA,MN,1),
     $  r2P5(MN),V5P(MN,MN),V5Pd(MN,MN),r2L5(MN),V5L(MN,MN)
	character*1 Atom

c---------------------------------------------------------
c read in potential data
c---------------------------------------------------------
	  r1cut=4.d0; r2cut=4.2d0; a0=0.529177d0; eV=27.2116d0
	  pi=acos(-1.d0); V=12345.d0; theta=-5.d0

	  read(IDpot,*) N1,N2
	  read(IDpot,*) (r1(i),i=1,N1)
	  read(IDpot,*) (r2(i),i=1,N2)

	  DO I=1,N1
	    if(r1(i).GT.r1cut) go to 1
	  END DO
1	  N1cut=I-1

	  DO I=1,N2
	    if(r2(i).GT.r2cut) go to 2
	  END DO
2	  N2cut=I-1
	  print*,'N1cut N2cut for 3D AV5Z',N1cut,N2cut,N1,N2

	  read(IDpot,*) ID0,vcc,vhf
	  read(IDpot,*) Atom, x1,y1; read(IDpot,*) Atom; read(IDpot,*) Atom,x2
	  r10=x2/a0; r20=sqrt((x1-x2)*(x1-x2)+y1*y1)/a0
          r30=sqrt(x1*x1+y1*y1)/a0
          theta0=(r10*r10+r20*r20-r30*r30)/(2.d0*r10*r20)
	  theta0=min(theta0,1.d0);theta0=max(theta0,-1.d0);theta0=acos(theta0)
	  Nt(1,1)=2
          V(2,1,1)=vcc/eV
          IF (abs(Vcc-12345.d0).gt.1.d0 .And. Vcc.le.-50.d0)
     $    V(2,1,1)=(vcc-vhf)*scale+vhf+V00
	  theta(2,1,1)=theta0
	  I0=1; J0=1

c read in potential
	  DO
	    read(IDpot,*,err=10) ID0,vcc,vhf
	    read(IDpot,*) Atom, x1,y1; read(IDpot,*) Atom; read(IDpot,*) Atom,x2

30	    r10=x2/a0; r20=sqrt((x1-x2)*(x1-x2)+y1*y1)/a0
            r30=sqrt(x1*x1+y1*y1)/a0
            theta0=(r10*r10+r20*r20-r30*r30)/(2.d0*r10*r20)
	    theta0=min(theta0,1.d0);theta0=max(theta0,-1.d0);theta0=acos(theta0)

	    do I=1,N1
	      if(abs(r10-r1(I)).le.1.d-4) go to 40
	    end do
	    print*,r10,ID0,vcc
	    stop 'cannot find r1'
40	    do j=1,N2
	      if(abs(r20-r2(j)).le.1.d-4) go to 50
	    end do
	    print*,r20,ID0,vcc
	    stop 'cannot find r2'
50	    continue
c	    print*,J,I

	    IF(J.ne.J0 .or. I.ne.I0) THEN
	      J0=J; I0=I
	      Nt(J0,I0)=2
	    ELSE
	      Nt(J0,I0)=Nt(J0,I0)+1
	    END IF

            V(Nt(J0,I0),j,i)=vcc/eV
            IF (abs(Vcc-12345.d0).gt.1.d0 .And. Vcc.le.-50.d0)
     $      V(Nt(J0,I0),j,i)=(vcc-vhf)*scale+vhf+V00
	    theta(Nt(J0,I0),J,I)=theta0

60	   END DO

10      continue

c read in additional 5Z potential for collinar geometry with F inbetween H2
	DO 
  	  read(IDpot,*,end=11) ID, vin1,vin2
          read(IDpot,*) r10,r20
	  IF(r20.ge.r10/2.d0) then
	    print*,r1,r20
	    stop 'Something wrong for F inbetween H2'
	  END IF

	  do i=1,N1
	   if (abs(r10-r1(i)).lt.1.d-4) go to 12
	  end do
	  print*,r10,r20
	  stop 'CANNOT find r1'
12	  i0=i

	  do j=1,N2
           if (abs(r20-r2(j)).lt.1.d-4) go to 13
          end do
          print*,r10,r20
          stop 'CANNOT find r2'
13        j0=j

          V(1,j0,i0)=Vin1/eV
          IF (abs(Vin1-12345.d0).gt.1.d-3 .And. Vin1.le.-50.d0) 
     $        V(1,j0,i0)=(vin2+scale*(vin1-vin2)+V00)
	    
	  theta(1,j0,i0)=0.d0
	END DO
11	close(IDpot)

c compare AVQZ and AV5Z PES for collinear geometry
	DO I=1,N1
          do i0=1,M1
           if (abs(r1(i)-s1(i0)).lt.1.d-4) go to 24
          end do
          print*,I,J,r1(i),r2(j)
          stop 'CANNOT FIND r1'
24        continue
	DO J=1,N2
          if(r1(i).LE.r1cut .or. r2(j).LE.r2cut) then
	  end if
	END DO
	
	END DO
	 
c substract AVQZ potential from AV5Z
	NC=0
	DO I=1,N1
	DO J=1,N2
          if(r1(i).GT.r1cut .and. r2(j).GT.r2cut) go to 199

	  do i0=1,M1
           if (abs(r1(i)-s1(i0)).lt.1.d-4) go to 14
          end do
          print*,I,J,r1(i),r2(j)
          stop 'CANNOT FIND r1'
14        continue

          do j0=1,M2
           if (abs(r2(j)-s2(j0)).lt.1.d-4) go to 15
          end do
          print*,I,J,r1(i),r2(j)
          stop 'CANNOT Find r2'
15        continue
	  V5L(j,i)=V5L(j,i)-V4(Nt4(j0,i0),j0,i0)

	  a0=0.529177d0;eV=27.2116d0
	  IST=2
	  if(abs(theta(1,j,i)+5.d0).gt.1.d-3) IST=1
	  do k=IST,Nt(J,I)
	    
	     call splint(Nt4(j0,i0),theta4(1,j0,i0),V4(1,j0,i0),V4d(1,j0,i0),
     $            theta(k,j,i),V4out)
	    dv=(V(k,j,i)-V4out)*27.2116d0
	     
	    if(abs(V(k,j,i)-12345.d0).gt.1.d-3) V(k,j,i)=V(k,j,i)-V4out
              
	  end do

199	  END DO
	  END DO
	  print*,'finished'

c	POTread5Z(IDpot,MN,N1,N2,r1,r2,MNA,Nt,theta,V,Vp,
c     $  scale,V00,N1cut,N2cut,M1,M2,s1,s2,Nt4,theta4,V4,V4d,
c     $  N2P5,r2P5,V5P,V5Pd,N2L5,r2L5,V5L)
c spline theta for 3D data
          DO i=1,N1
          DO j=1,N2
            if(r1(i).GT.r1cut .and. r2(j).GT.r2cut) go to 200

c calculate first order derivative from V2d
            dvdt=0.d0
	    theta0=acos(min(r1(i)/(2.d0*r2(j)),1.d0-1.d-20))
            IF (theta0.gt.1.d-2 .or. abs(r1(i)-2.d0*r2(j)).le.1.d-5) THEN
              theta(1,j,i)=theta0
              Dist=r2(j)*sin(theta(1,j,i))
              CALL SPLINTp(N2P5,r2P5,V5P(1,i),V5Pd(1,i),Dist,Esymm,Epsymm)
              dvdt=Epsymm*cos(theta(1,j,i))*r2(j)
              V(1,j,i)=Esymm
            END IF

c add in AV5Z data for collinear geometry with F at one end.
	    Nt(j,i)=Nt(j,i)+1; theta(Nt(j,i),j,i)=pi
	    if(abs(r2L5(j)-r2(j)).gt.1.d-3) then
	      print*,'stop 5Z collinear geometry'
	      print*,r1(i),r2L5(j),r2(j)
	      stop
	    end if
	    V(Nt(j,i),j,i)=V5L(j,i)

            CALL spline(Nt(j,i),theta(1,j,i),V(1,j,i),dvdt,0.d0,Vp(1,j,i))

200       END DO
          END DO

	RETURN
	END

C##################################################################
	subroutine POTread3d(IDpot,MN,N1,N2,r1,r2,MNA,Nt,theta,V,Vp,
     $  scale,V00,N1cut,N2cut,N2P4,r2P4,V4P,V4Pd)
	implicit real*8 (a-h,o-z)
	dimension V(MNA,MN,MN),Vp(MNA,MN,MN),r1(1),r2(1),theta(MNA,MN,MN),
     $  V1d(MN),X0(MN),WK(MN),F(2),Nt(MN,1),Ed(2),
     $  V4P(MN,MN),V4Pd(MN,MN),r2P4(MN)
	character*1 Atom

c---------------------------------------------------------
c read in potential data
c---------------------------------------------------------
	  r1cut=4.d0; r2cut=4.2d0; a0=0.529177d0; eV=27.2116d0
	  pi=acos(-1.d0); V=12345.d0

c	  open(1,file='POT.dat',status='old')
	  read(IDpot,*) N1,N2
	  read(IDpot,*) (r1(i),i=1,N1)
	  read(IDpot,*) (r2(i),i=1,N2)

	  DO I=1,N1
	    if(r1(i).GT.r1cut) go to 1
	  END DO
1	  N1cut=I-1

	  DO I=1,N2
	    if(r2(i).GT.r2cut) go to 2
	  END DO
2	  N2cut=I-1

	  IF(N2.GT.MN .OR. N1.GT.MN ) THEN
	    print*,'increase MN'; stop
	  END IF
	  print*,'N1cut N2cut for 3D AVQZ',N1cut,N2cut,N1,N2

	  dtt=1.d0/180.d0*pi;Nall=0.d0
c read in potential
	  DO
	    read(IDpot,*,end=10) ID0,vcc,vhf
	    read(IDpot,*) Atom, x1,y1; read(IDpot,*) Atom; read(IDpot,*) Atom,x2

30	    r10=x2/a0; r20=sqrt((x1-x2)*(x1-x2)+y1*y1)/a0
            r30=sqrt(x1*x1+y1*y1)/a0
            theta0=(r10*r10+r20*r20-r30*r30)/(2.d0*r10*r20)
	    theta0=min(theta0,1.d0);theta0=max(theta0,-1.d0);theta0=acos(theta0)

	    do I=1,N1
	      if(abs(r10-r1(I)).le.1.d-4) go to 40
	    end do
	    stop 'cannot find r1'
40	    do j=1,N2
	      if(abs(r20-r2(j)).le.1.d-4) go to 50
	    end do
	    stop 'cannot find r2'
50	    continue
c	    print*,J,I

	    Nt(J,I)=1
            V(1,j,i)=vcc/eV
            IF (abs(Vcc-12345.d0).gt.1.d0 .And. Vcc.le.-50.d0)
     $      V(1,j,i)=(vcc-vhf)*scale+vhf+V00
	    theta(1,j,i)=theta0

	    do k=2,19
	      read(IDpot,*,end=10) ID0,vcc,vhf
	      read(IDpot,*) Atom, x1,y1; read(IDpot,*) Atom; read(IDpot,*) Atom,x2
	      r10=x2/a0; r20=sqrt((x1-x2)*(x1-x2)+y1*y1)/a0
              r30=sqrt(x1*x1+y1*y1)/a0
              theta1=(r10*r10+r20*r20-r30*r30)/(2.d0*r10*r20)
	      theta1=min(theta1,1.d0);theta1=max(theta1,-1.d0);theta1=acos(theta1)

	      if(abs(r10-r1(i)).gt.1.d-2 .or. abs(r20-r2(j)).gt.1.d-2) then
	        print*,'something is wrong in reading', j,i,k
	        go to 30
	      end if

	      if(theta1.lt.theta0) then
	         if(theta1.lt.theta(1,j,i)) then
                   Ed(1)=vcc/eV
                   IF (Vcc.le.-50.d0) Ed(1)=(vcc-vhf)*scale+vhf+V00
	           Nst=2
	           go to 20
	         else
	           go to 60
	         end if
	      end if

	      theta0=theta1

	      if(abs(Vcc-12345.d0).gt.1.d0) then  ! converged energy
	        Nt(j,i)=Nt(j,i)+1
	        theta(Nt(j,i),j,i)=theta1
                V(Nt(j,i),j,i)=vcc/eV
                IF (Vcc.le.-50.d0) V(Nt(j,i),j,i)=(vcc-vhf)*scale+vhf+V00
	      end if

	      if(abs(pi-theta1).le.1.d-2) then
	        Nst=1
	        go to 20
	      end if

	    end do   ! end of 19

c read in data for energy derivative
	   Nst=1
20         do k=Nst,2
              read(IDpot,*,end=10) ID0,vcc,vhf
              read(IDpot,*) Atom, x1,y1; read(IDpot,*) Atom; read(IDpot,*) Atom,x2

              r10=x2/a0; r20=sqrt((x1-x2)*(x1-x2)+y1*y1)/a0

              r30=sqrt(x1*x1+y1*y1)/a0
              theta0=(r10*r10+r20*r20-r30*r30)/(2.d0*r10*r20)
	      theta0=min(theta0,1.d0);theta0=max(theta0,-1.d0);theta0=acos(theta0)

              if(abs(r10-r1(i)).gt.1.d-2 .or. abs(r20-r2(j)).gt.1.d-2) go to 30

              Ed(k)=vcc/eV
              IF (Vcc.le.-50.d0) Ed(k)=(vcc-vhf)*scale+vhf+V00
            end do

c	    dvdt(j,i)=(Ed(2)-Ed(1))/dtt
c	    print*,'first order derivative',j,i,dvdt(j,i)

60	   END DO

10	close(IDpot)

c spline theta for 3D data
          DO i=1,N1
          DO j=1,N2
            if(r1(i).GT.r1cut .and. r2(j).GT.r2cut) go to 200

c calculate first order derivative from V2d
            dvdt=0.d0
            IF (theta(1,j,i).gt.1.d-2) THEN
              Dist=r2(j)*sin(theta(1,j,i))
              CALL SPLINTp(N2P4,r2P4,V4P(1,i),V4Pd(1,i),Dist,Esymm,Epsymm)
              dvdt=Epsymm*cos(theta(1,j,i))*r2(j)
              V(1,j,i)=Esymm
            END IF

            CALL spline(Nt(j,i),theta(1,j,i),V(1,j,i),dvdt,0.d0,Vp(1,j,i))

200       END DO
          END DO


	RETURN
	END

c#####################################################################
	subroutine POTread2D(IDpot,MN,N1,N2,r1,r2,V,Vp,scale,V00,dvdt0,
     $  N1p,N2p,r1p,r2p,V4P)
	implicit real*8 (a-h,o-z)
	dimension V(MN,1),Vp(MN,1),r1(1),r2(1),V1d(MN),X0(MN),WK(MN),
     $  r1p(MN),r2p(MN),V4P(MN,1)

	dvdt2=0.d0
	N1=1; N2=1; IDcheck=0; eV=27.2116d0
	read(IDpot,*) ID, Vin1,Vin2
	read(IDpot,*) r1(1),r2(1)
	V(1,1)=(vin2+scale*(vin1-vin2)+V00)
	DO
	  read(IDpot,*,end=2) ID, vin1,vin2
	  read(IDpot,*) r10,r20

	  IF (abs(r10-r1(N1)).gt.1.d-4) THEN
	    N1=N1+1; N2=1; r1(N1)=r10; IDcheck=1
	  ELSE
	    N2=N2+1
	    IF (IDcheck.Eq.0) then
	      r2(N2)=r20
	    ELSE
	      IF (abs(r20-r2(N2)).gt.1.d-4) then
                print*,'Something is wroing in POTread2D'
	        print*,vin1,vin2
	 	print*,r10,r20
	        stop
	      END IF
	    END IF
	  END IF

	  V(N2,N1)=Vin1/eV
          IF (abs(Vin1-12345.d0).gt.1.d-3 .And. Vin1.le.-50.d0) 
     $	      V(N2,N1)=(vin2+scale*(vin1-vin2)+V00)

	END DO
2	close(IDpot)       
	print*,N1,N2

c check potential for big r1,r2 values
	DO I=1,N1
	  
	  N2c=N2
	  DO J=N2-1,1,-1
	    if(abs(V(J,I)-V(N2,I)).lt.1.d-8 .and. 
     $         abs(V(J,I)-12345.d0/eV).gt.1.d-3) N2c=J
	  END DO

	  DO J=N2c,N2-1
	    V(J,I)=V(N2,I)
	  END DO
	END DO

c do spline for missing points
	DO I=1,N1

	N2in=0
	DO j=1,N2
	  IF (abs(V(j,i)-12345.d0/eV).GT.1.d0) then
	     N2in=N2in+1
	     V1d(N2in)=V(j,i)
	     X0(N2in)=r2(j)

	     IF(mod(IDpot,100).ne.0) then

c substract reference potential (N1p,N2p,r1p,r2p,V4P)
	     do ip=1,N1p
	       if(abs(r1(i)-r1p(ip)).le.1.d-5) go to 211
	     end do
	     print*,'Cannot find r1', r1(i),r2(j)
	     stop '2110'
211	     continue

	     do jp=1,N2p
               if(abs(r2(j)-r2p(jp)).le.1.d-5) go to 212
             end do
             print*,'Cannot find r2', r1(i),r2(j)
             stop '2120'
212          continue
	     V1d(N2in)=V1d(N2in)-V4P(jp,ip)

	     END IF

	  END IF
	END DO

c do cut-off for points at beginning
	if (mod(IDpot,100).ne.0) then
	  DO k=1,N2in
	    if(abs(V1d(k)).le.0.05d0) go to 213
	  END DO
213	  k0=k-1
	  dvdr=(V1d(k0+2)-V1d(k0+1))/(x0(k0+2)-x0(k0+1))
	  DO k=1,k0
	    V1d(k)=V1d(k0+1)+dvdr*(x0(k)-x0(k0+1))
	  END DO
	END IF
	    
	dvdt1=dvdt0
	IF (abs(dvdt0-1.d0).le.1.d-5)
     $  dvdt1=(V1d(2)-V1d(1))/(x0(2)-x0(1))
	CALL spline(N2in,X0,V1d,dvdt1,dvdt2,WK)

	DO j=1,N2
	  r2in=r2(j)
          CALL splint(N2in,X0,V1d,WK,r2in,Vout)
	  V(j,i)=Vout
	END DO

	END DO  ! end of spline

	do j=1,N2
	  if(r2(j).gt.5.d0) go to 123
	end do
123	N2s=j-1


	RETURN
	END
	
C##################################################################
C# SPLINE ROUTINES
C#            Numerical recipes in fortran
C#            Cambrige University Press
C#            York, 2nd edition, 1992.
C##################################################################
      SUBROUTINE splint(n,xa,ya,y2a,x,y)
      implicit double precision  (a-h,o-z)
      DIMENSION xa(n),y2a(n),ya(n)
      klo=1
      khi=n
 1    if (khi-klo.gt.1) then
        k=(khi+klo)/2
        if(xa(k).gt.x)then
          khi=k
        else
          klo=k
        endif
      goto 1
      endif
      h=xa(khi)-xa(klo)
      if (h.eq.0.0d0) then
	write(*,*)' error h=',0
	write(*,*)' number of point=',n
	return
	endif
      a=(xa(khi)-x)/h;a2=a*a
      b=(x-xa(klo))/h;b2=b*b
      y=a*ya(klo)+b*ya(khi)+((a2*a-a)*y2a(klo)+(b2*b-b)*y2a(khi))*(h*h)/6.0d0
      return
      END

c      a=(xa(khi)-x)/h
c      b=(x-xa(klo))/h
c      y=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**
c     &2)/6.0d0

C##############################################################################
      SUBROUTINE spline(n,x,y,yp1,ypn,y2)
      implicit double precision  (a-h,o-z)
      DIMENSION x(n),y(n),y2(n)
      PARAMETER (NMAX=100)
      DIMENSION u(NMAX)
      if (yp1.gt..99d30) then
        y2(1)=0.0d0
        u(1)=0.0d0
      else
        y2(1)=-0.5d0
        u(1)=(3.0d0/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
      endif
      do 11 i=2,n-1
        sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
        p=sig*y2(i-1)+2.0d0
        y2(i)=(sig-1.0d0)/p
        u(i)=(6.0d0*((y(i+1)-y(i))/(x(i+
     *1)-x(i))-(y(i)-y(i-1))/(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*
     *u(i-1))/p
11    continue
      if (ypn.gt..99d30) then
        qn=0.0d0
        un=0.0d0
      else
        qn=0.5d0
        un=(3.0d0/(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
      endif
      y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.0d0)
      do 12 k=n-1,1,-1
        y2(k)=y2(k)*y2(k+1)+u(k)
12    continue
      return
      END


C##################################################################
      SUBROUTINE splintp(n,xa,ya,y2a,x,y,yp)
      implicit double precision  (a-h,o-z)
      DIMENSION xa(n),y2a(n),ya(n)
      klo=1
      khi=n
 1    if (khi-klo.gt.1) then
        k=(khi+klo)/2
        if(xa(k).gt.x)then
          khi=k
        else
          klo=k
        endif
      goto 1
      endif
      h=xa(khi)-xa(klo)
      if (h.eq.0.0d0) then
        write(*,*)' error h=',0
        write(*,*)' number of point=',n
        return
      endif
      a=(xa(khi)-x)/h;a2=a*a
      b=(x-xa(klo))/h;b2=b*b
      y=a*ya(klo)+b*ya(khi)+((a2*a-a)*y2a(klo)+(b2*b-b)*y2a(khi))*(h*h)/6.0d0
      yp=(ya(khi)-ya(klo))/h+h*(y2a(khi)*(3.d0*b2-1.d0)
     $-y2a(klo)*(3.d0*a2-1.d0))/6.d0
      return
      END
C##############################################################################
c
c  _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
c  _/          surface950621 H3 surface evaluation routines          _/
c  _/              (also known as the bkmp2 surface)                 _/
c  _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
      subroutine surfid(h3id)
c-----------------------------------------------------------------
c  h3 surface identification number (based on date of fitting)
c-----------------------------------------------------------------
      implicit double precision (a-h,o-z)
	h3id = 950621.0
      return
      end
c
c      subroutine bkmp2(r,Vtot,dVtot,id)
      subroutine bkmp2(r,Vtot)
c-----------------------------------------------------------------
c Calculate total H3 potential from all of its parts
c if (id.gt.0) also calculate the dV/dr derivatives
c all distances are in bohrs and all energies are in hartrees
c
c For a discussion of this surface, see:
c    A.I.Boothroyd, W.J.Keogh, P.G.Martin, M.R.Peterson
c    Journal of Chemical Physics 95 pp4343-4359 (Sept15/91)
c    and JCP 104 pp 7139-7152 (May8/96)
c
c Note: this file contains parameters for a surface refitted
c       on june21/95 to a set of several thousand ab initio
c       points.  the '706' surface parameters have been
c       commented out.  The routine names have been modified
c       slightly (usually a '95' appended) so that a programme
c       could easily call and compare both of our surfaces.
c
c Note: the surface parameter values as published lead to an
c       anomolously deep van der Waals well for a very compact
c       H2 molecule (say r=0.8).  After that paper was submitted,
c       this problem was fixed and the corrected Cbend coefficients
c       are used in this version of the surface (version 706).
c       The old coefficients are still in subr.vbcb but have been
c       commented out.
c
c any QUESTIONS/PROBLEMS/COMMENTS concerning this programme can be
c addressed to :    wkeogh@alchemy.chem.utoronto.ca
c (if necessary, via   boothroy@cita.utoronto.ca    boothroy@utordop.bitnet
c                 or   pgmartin@cita.utoronto.ca    pgmartin@utordop.bitnet )
c
c version:
c apr12/95 ... parameters for surface850308 added
c jul27/91 ... surf706d.out Cbend values put in
c----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      dimension r(3),dVtot(3)
      dimension dVlon(3),dVas(3),dVbnda(3),dVbndb(3),
     .          dCal(3), dCas(3),dCbnda(3),dCbndb(3)
      dimension vb(2,25),cb(2,25)
c     dimension vbp(2,25),cbp(2,25)
c     dimension dB1a(3),dB1b(3)
      dimension dT(3),T(3)
c     dimension dB2(3),dB3(6),expass(4)
      ipr = 0

	id=0
      call chgeom(r,ivalid)
      if(ivalid.lt.1)then
         vtot     = 99.0
         dvtot(1) = 0.d0
         dvtot(1) = 0.d0
         dvtot(1) = 0.d0
c        write(6,6999) vtot,dvtot
         return
      end if
 6999 format('invalid geometry, v,dv=',4f8.2)
c
c
c  zero everything to avoid any 'funny' values:
      Vlon  = 0.d0
      Vas   = 0.d0
      Vbnda = 0.d0
      Vbndb = 0.d0
      Cal   = 0.d0
      Cas   = 0.d0
      cbnda = 0.d0
      cbndb = 0.d0
      do i=1,3
	 dVlon(i) = 0.d0
	 dVas(i)  = 0.d0
	 dVbnda(i)= 0.d0
	 dVbndb(i)= 0.d0
	 dCal(i)  = 0.d0
	 dCas(i)  = 0.d0
	 dCbnda(i)= 0.d0
	 dCbndb(i)= 0.d0
      end do
c  zero the vb and cb arrays (used only for numerical derivatives)
      do i=1,25
	 vb(1,i) = 0.d0
	 vb(2,i) = 0.d0
	 cb(1,i) = 0.d0
	 cb(2,i) = 0.d0
      end do
      call h3lond95( r, Vlon, dVlon )
      call vascal95( r, Vas, dVas )
c    now do any corrections required for compact geometries:
      call compac95(r,icompc,T,dT)
c    oct.3/90 compact routines only called for compact geometries:
      if(icompc.ge.1)then
         call csym95  ( r, cal, dCal )
         call casym95 ( r, Cas, dCas,  1 ,T,dT)
      end if
      call vbcb95(r,icompc,T,dT,ipr,
     .          Vbnda,Vbndb,dVbnda,dVbndb,
     .          Cbnda,Cbndb,dCbnda,dCbndb)
c    add up the various parts of the potential:
      Vtot = Vlon + Vas + Vbnda + Vbndb + Cal + Cas + Cbnda + Cbndb 
     .       + 0.1744957722d0
c    add up the various parts of the derivative:
      dVtot(1) =   dVlon(1) + dVas(1) + dVbnda(1) + dVbndb(1)
     .           +  dCal(1) + dCas(1) + dCbnda(1) + dCbndb(1)
      dVtot(2) =   dVlon(2) + dVas(2) + dVbnda(2) + dVbndb(2)
     .           +  dCal(2) + dCas(2) + dCbnda(2) + dCbndb(2)
      dVtot(3) =   dVlon(3) + dVas(3) + dVbnda(3) + dVbndb(3)
     .           +  dCal(3) + dCas(3) + dCbnda(3) + dCbndb(3)

      if(ipr.gt.0) then
	  write(7,*)
	  write(7,*) 'h3tot enter --------------------------------'
          write(7,7000) r,icompc,vtot,
     .             vlon,vas,cal,cas,vbnda,vbndb,cbnda,cbndb
	  if(ipr.gt.1)then
		   write(7,7100) dVtot,dVlon,dVas,dVbnda,dVbndb,
     .                           dCal,dCas,dCbnda,dCbndb
	  end if
	  write(7,7400) (vb(1,i),i=1,5),(vb(2,i),i=1,5)
	  write(7,7500) (cb(1,i),i=1,9),(cb(2,i),i=1,9)
          write(7,7999) Vbnda,Vbndb,Cbnda,Cbndb,
     .                  dVbnda,dvbndb,dCbnda,dCbndb
	  write(7,*) 'exiting subr.h3tot'
	  write(7,*) 'h3tot exit ---------------------------------'
      end if
      return
 7400 format('Vba values: ',5(1x,g12.6),/,'Vbb values: ',5(1x,g12.6))
 7500 format('Cba values: ',3(1x,f16.8),/,
     .       '            ',3(1x,f16.8),/,
     .       '            ',3(1x,f16.8),/,
     .       'Cbb values: ',3(1x,f16.8),/,
     .       '            ',3(1x,f16.8),/,
     .       '            ',3(1x,f16.8))
 7000 format(5x,'    r =',3(1x,f16.10),' icompac = ',i1,/,
     .       5x,'Vtot  = ',f18.12,/,
     .       5x,'Vlon  = ',f18.12,'       Vas   = ',g18.12,/,
     .       5x,'Cal   = ',g18.12,'       Cas   = ',g18.12,/,
     .       5x,'Vbnda = ',g18.12,'       Vbndb = ',g18.12,/,
     .       5x,'Cbnda = ',g18.12,'       Cbndb = ',g18.12)
 7100 format(' dVtot   = ',3(1x,g18.12),/,
     .       ' dVlon   = ',3(1x,g18.12),/,
     .       ' dVas    = ',3(1x,g18.12),/,
     .       ' dVbnda  = ',3(1x,g18.12),/,
     .       ' dVbndb  = ',3(1x,g18.12),/,
     .       ' dCal    = ',3(1x,g18.12),/,
     .       ' dCas    = ',3(1x,g18.12),/,
     .       ' dCbnda  = ',3(1x,g18.12),/,
     .       ' dCbndb  = ',3(1x,g18.12))
 7750 format(3(1x,g16.10))
 7751 format(/,3(1x,g16.10))
 7999 format('   Vbnda        Vbndb        Cbnda        Cbndb ',/,
     .        4(e12.6,2x),/,
     .       'dVbnda: ',3(e16.10,1x),/,
     .       'dVbndb: ',3(e16.10,1x),/,
     .       'dCbnda: ',3(e16.10,1x),/,
     .       'dCbndb: ',3(e16.10,1x))
      end
c
      subroutine triplet95(r,E3,id)
C--------------------------------------------------------------------
c apr12/95 surface950308 values added
c oct04/90 surface626 values added
C  H2 triplet curve and derivatives:
c  calculates triplet potential and first derivative
c  uses truhlar horowitz equation with our extension
c  uses the Johnson correction at short distances (r < rr)
c     if r .ge. rr         use modified t/h triplet equation
c     if r .le. rl         use the Johnson correction
c     in between           use the transition equation
C--------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      dimension E3(3),E1(3)
      parameter( rl=0.95d0, rr=1.15d0 )
      parameter( z1=1.d0, z2=2.d0, z4=4.d0)
c triplet and johnson values mar08/95

c     parameter(
c    .  a1= -0.0460730469,a2=-16.3948753611,a3=-27.6026451962,
c    .  a4=  2.0383359014,a5= -6.8887433326,a6=  1.6216005938,
c    .  C1= -0.4110341110310669,C2= -0.0767398928636804,
c    .  C3=  0.4302169435606523)
c triplet values jun21

      parameter(
     .  a1= -0.0298546962,a2=-23.9604445036,a3=-42.5185569474,
     .  a4=  2.0382390988,a5=-11.5214861455,a6=  1.5309487826,
     .  C1= -0.4106358351531854,C2= -0.0770355790707090,
     .  C3=  0.4303193846943223) !jun21 fit
c  triplet and johnson values from fit621:
c     parameter(a1=-0.0253496194,a2=-29.2302126444,
c    .          a3=-50.7225015503,a4= 2.0452676876 ,
c    .          a5=-12.2408908509,a6= 1.6733157383 )
c     parameter( C1=-0.4170298146519658,
c    .           C2=-0.0746027774843370,C3= 0.4297899952237434 )
c surface parameters from fit601.out
c     parameter(a1=-0.66129429,a2=-1.99434198,a3=-2.37604328,
c    .          a4= 2.08107802,a5=-0.0313032510,a6=3.76546699,
c    .          C1=-0.4222590135447196,C2=-0.0731117796738824,
c    .          C3= 0.4295918082189010 )
      ipr = 0
      E3(3) = 0.d0
      if(r.ge.rr )then
c       modified truhlar/horowitz triplet equation:
         exdr = dexp( -a4*r )
         rsq  = r*r
         ra6  = r**(-a6)
         E3(1) = a1* ( a2 + r + a3*rsq + a5*ra6 )*exdr
c       first derivative of triplet curve:
         ra61 = r**(-a6-z1)
         E3(2) = a1*exdr*
     .   ( z1 -a2*a4 +(z2*a3-a4)*r -a3*a4*rsq -a5*a6*ra61 -a4*a5*ra6 )
      end if
      if(r.lt.rr )then
          dr = r- rl
          call vh2opt95(r,e1,2)
	  if( r.le.rl ) then
c          Johnson triplet equation:
            E3(1) = E1(1) + c2*dr + c3
	    E3(2) = E1(2) + c2
          else
c          Transition equation:
            E3(1) = E1(1) + c1*dr*dr*dr + c2*dr + c3
	    E3(2) = E1(2) + 3.d0*c1*dr*dr + c2
	  end if
      end if
      if(ipr.gt.0) write(7,7100) e3
 7100 format('  triplet:  E3 = ',3(1x,f12.8))
      return
      end
C
c
      subroutine vH2opt95(r,E,ideriv)
C-----------------------------------------------------------------
c jul09/90 ... super duper speedy version
C self-contained version of schwenke's H2 potential
C all distances in bohrs and all energies in Hartrees
C (1st deriv added on May 2 1989; 2nd deriv on May 28 1989)
C-----------------------------------------------------------------
      implicit double precision (a-h,o-z)
      dimension E(3)
      parameter(a0=0.03537359271649620,    a1= 2.013977588700072    ,
     .  a2= -2.827452449964767        ,    a3= 2.713257715593500    ,
     .  a4= -2.792039234205731        ,    a5= 2.166542078766724    ,
     .  a6= -1.272679684173909        ,    a7= 0.5630423099212294   ,
     .  a8= -0.1879397372273814       ,    a9= 0.04719891893374140  ,
     . a10= -0.008851622656489644     ,   a11= 0.001224998776243630 ,
     . a12= -1.227820520228028d-04    ,   a13= 8.638783190083473d-06,
     . a14= -4.036967926499151d-07    ,   a15= 1.123286608335365d-08,
     . a16= -1.406619156782167d-10 )
      parameter( r0=3.5284882d0,  DD=0.160979391d0,
     .           c6=6.499027d0,   c8=124.3991d0,    c10=3285.828d0)
c     Ediss = 0.174445d0
      E(1)  = -999.d0
      E(2)  = -999.d0
      E(3)  = -999.d0
      r2  = r  *r
      r3  = r2 *r
      r4  = r3 *r
      r5  = r4 *r
      r6  = r5 *r
      r7  = r6 *r
      r8  = r7 *r
      r9  = r8 *r
      r10 = r9 *r
      r11 = r10*r
      r12 = r11*r
      r13 = r12*r
      r14 = r13*r
      r15 = r14*r
      r02 = r0*r0
      r04 = r02*r02
      r06 = r04*r02
      rr2  = r2 + r02
      rr4  = r4 + r04
      rr6  = r6 + r06
      rr25 = rr2*rr2*rr2*rr2*rr2
c     general term:  a(i)*r(i-1),  i=0,16
      alphaR =  a0/r + a1
     .            + a2 *r   + a3 *r2  + a4 *r3  + a5 *r4  + a6 *r5
     .            + a7 *r6  + a8 *r7  + a9 *r8  + a10*r9  + a11*r10
     .            + a12*r11 + a13*r12 + a14*r13 + a15*r14 + a16*r15
      exalph = dexp(alphaR)
      vsr = DD*(exalph-1.d0)*(exalph-1.d0) - DD
      vlr =  -C6/rr6   -C8/(rr4*rr4)   -C10/rr25
      E(1)=  vsr +  vlr
c
C  calculate first derivative if required:
      if(ideriv.ge.1)then
         r3 = r2*r
         r5 = r4*r
         rr26 = rr25*rr2
         rr43 = rr4*rr4*rr4
c        general term:  (i-1)*a(i)*r**(i-2)  ,  i=0,16
	 dalphR = -a0/r2 + a2 + 2.d0*a3*r
     .          +  3.d0 *a4 *r2  +  4.d0 *a5 *r3  +  5.d0 *a6 *r4
     .          +  6.d0 *a7 *r5  +  7.d0 *a8 *r6  +  8.d0 *a9 *r7
     .          +  9.d0 *a10*r8  + 10.d0 *a11*r9  + 11.d0 *a12*r10
     .          + 12.d0 *a13*r11 + 13.d0 *a14*r12 + 14.d0 *a15*r13
     .          + 15.d0 *a16*r14
	 dvsr = 2.d0 *DD *(exalph-1.d0) *exalph *dalphR
	 dvlr =    6.d0*C6*r5 / (rr6*rr6)
     .          +  8.d0*C8*r3 / rr43  + 10.d0*C10*r / rr26
	 E(2) = dvsr + dvlr
      end if
c
C  calculate second derivative if required:
      if(ideriv.ge.2)then
         r10  = r6*r4
         rr27 = rr26*rr2
         rr44 = rr43*rr4
         rr62 = rr6*rr6
         rr63 = rr62*rr6
c        general term: (i-1)*(i-2)*a(i)*r**(i-3),    i=0,16
	 ddalph = 2.d0*a0/r3 +  2.d0*a3      +  6.d0*a4 *r
     .     + 12.d0*a5 *r2    + 20.d0*a6 *r3  + 30.d0*a7 *r4
     .     + 42.d0*a8 *r5    + 56.d0*a9 *r6  + 72.d0*a10*r7
     .     + 90.d0*a11*r8    +110.d0*a12*r9  +132.d0*a13*r10
     .     +156.d0*a14*r11   +182.d0*a15*r12 +210.d0*a16*r13
	 ddvsr = 2.d0 *DD *exalph
     .   *( (2.d0*exalph-1.d0)*dalphR*dalphR + (exalph-1.d0)*ddalph )
	 ddvlr =- 72.d0* C6 *r10 / rr63 -96.d0 *C8 *r6 / rr44
     .		-120.d0*C10 *r2  / rr27 +30.d0 *C6 *r4 / rr62
     .		+ 24.d0* C8 *r2  / rr43 +10.d0 *C10    / rr26
	 E(3) = ddvsr + ddvlr
      end if
      return
      end
c
      subroutine h3lond95(r,Vlon,dVlon)
c----------------------------------------------------------------------
c version of may 12/90  ... derivatives corrected (0.5 changed to 0.25)
c calculates the h3 london terms and derivatives
c modified oct 7/89 to include eps**2 term which rounds off the
c cusp in the h3 potential which occurs at equilateral triangle
c configurations
c----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      real*8 Q(3),J(3),Jt
      dimension r(3),e1(3),e3(3),esing(3),etrip(3),dVlon(3)
      dimension de1(3),de3(3)
c     dimension dJ1(3),dJ2(3),dJ3(3)
      parameter( half=0.5d0, two=2.d0 , eps2=1.d-12 )
      ipr = 0
      do i=1,3
         call vh2opt95(r(i),esing,2)
	  e1(i) = esing(1)
	 de1(i) = esing(2)
	 call triplet95(r(i),etrip,2)
	 if(ipr.gt.0) write(7,7400) i,esing,etrip
	  e3(i) = etrip(1)
	 de3(i) = etrip(2)
         q(i)  = half*(E1(i) + E3(i))
         j(i)  = half*(E1(i) - E3(i))
      end do
      sumQ  =   q(1) + q(2) + q(3)
      sumJ  =   dabs( j(2)-j(1) )**2
     .        + dabs( j(3)-j(2) )**2
     .        + dabs( j(3)-j(1) )**2
      Jt     = half*sumJ + eps2
      rootJt = dsqrt(Jt)
      Vlon   = sumQ - rootJt
      if(ipr.gt.0) then
	 write(7,7410) sumq,sumj
         write(7,7420) vlon,rootJt
      end if
c  calculate the derivatives with respect to r(i):
      dVlon(1) = half*(dE1(1)+de3(1))
     .         - 0.25d0*(two*j(1)-j(2)-j(3))*(dE1(1)-de3(1))/rootJt
      dVlon(2) = half*(dE1(2)+de3(2))
     .         - 0.25d0*(two*j(2)-j(3)-j(1))*(dE1(2)-de3(2))/rootJt
      dVlon(3) = half*(dE1(3)+de3(3))
     .         - 0.25d0*(two*j(3)-j(1)-j(2))*(dE1(3)-de3(3))/rootJt
      if(ipr.gt.0) then
         write(7,7000) r,e1,e3,vlon
	 write(7,7100) q,j
	 write(7,7200) dVlon
      end if
 7000 format('             r = ',3(1x,f12.6),/,
     .       '            E1 = ',3(1x,f12.8),/,
     .       '            E3 = ',3(1x,f12.8),/,
     .       '          vlon = ',1x,f12.8)
 7100 format(13x,'q = ',3(1x,e12.6),/,13x,'j = ',3(1x,e12.6))
 7200 format('         dVlon = ',3(1x,g12.6))
 7400 format('from subr.london: ',/,
     .       '  using r',i1,':   Esinglet=',3(1x,f12.8),/,
     .       '         ',1x,'    Etriplet=',3(1x,f12.8))
 7410 format('         sumQ = ',g12.6,'        sumJ = ',g12.6)
 7420 format('         Vlon = ',f12.8,'      rootJt = ',g12.6)
      return
      end
c
      subroutine Vascal95(rpass,Vas,dVas)
c------------------------------------------------------------------
c version of apr12/95  950308 values
c version of oct11/90  fit632c.out values
c version of oct5/90   surf626 values
c  calculate the asymmetric correction term and its derivatives
c  see equations [14] to [16] of truhlar/horowitz 1978 paper
c------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      dimension dVas(3),dA(3),dS(3),rpass(3)
c vasym values jun21

      parameter(
     . aa1=0.3788951192E-02, aa2=0.1478100901E-02,
     . aa3=-.1848513849E-03, aa4=0.9230803609E-05,
     . aa5=-.1293180255E-06, aa6=0.5237179303E+00,
     . aa7=-.1112326215E-02) !jun21 fit
c--- vasym values mar08.95

c     parameter(
c    . aa1=0.3759731624E-02, aa2=0.1476254095E-02,
c    . aa3=-.1866759453E-03, aa4=0.9218646237E-05,
c    . aa5=-.1287906069E-06, aa6=0.5201790843E+00,
c    . aa7=-.1062909514E-02)
c--- Vasym values from fit632c ----------------------
c     parameter(
c    . aa1=0.3438222224E-02,aa2=0.1398145763E-02,
c    . aa3=-.1923999449E-03,aa4=0.9712737075E-05,
c    . aa5=-.1263794562E-06,aa6=0.5181432712E+00,
c    . aa7=-.9487002995E-03 )
      ipr = 0
      r1 = rpass(1)
      r2 = rpass(2)
      r3 = rpass(3)
      R  = r1 + r2 + r3
      Rsq = R*R
      Rcu = Rsq*R
C   calculate the Vas term first (eq.14 of truhlar/horowitz)
      call acalc95(r1,r2,r3,A,dA)
        A2 = A *A
        A3 = A2*A
        A4 = A3*A
        A5 = A4*A
        exp1 = dexp(-aa1*Rcu)
	exp6 = dexp(-aa6*R)
        S = aa2*A2 + aa3*A3 + aa4*A4 + aa5*A5
      Vas = S*exp1 +  aa7*A2 *exp6 / R
         dS(1)  = ( 2.d0*aa2*A  +3.d0*aa3*A2
     .             +4.d0*aa4*A3 +5.d0*aa5*A4) * dA(1)
         dVas(1) =  -3.d0*aa1*Rsq*S*exp1 + dS(1)*exp1
     .             -aa7*A2*exp6/Rsq  + 2.d0*aa7*A*dA(1)*exp6/R
     .             -aa6*aa7*A2*exp6/R
         dS(2)  = ( 2.d0*aa2*A  +3.d0*aa3*A2
     .             +4.d0*aa4*A3 +5.d0*aa5*A4) * dA(2)
         dVas(2) =  -3.d0*aa1*Rsq*S*exp1 + dS(2)*exp1
     .             -aa7*A2*exp6/Rsq  + 2.d0*aa7*A*dA(2)*exp6/R
     .             -aa6*aa7*A2*exp6/R
         dS(3)  = ( 2.d0*aa2*A  +3.d0*aa3*A2
     .             +4.d0*aa4*A3 +5.d0*aa5*A4) * dA(3)
         dVas(3) =  -3.d0*aa1*Rsq*S*exp1 + dS(3)*exp1
     .             -aa7*A2*exp6/Rsq  + 2.d0*aa7*A*dA(3)*exp6/R
     .             -aa6*aa7*A2*exp6/R
      if(ipr.gt.0) write(7,7100) Vas,dS,dVas
 7100 format('    from subr.vascal:   Vas  = ',  1x,g12.6, /,
     .       '                        dS   = ',3(1x,g12.6),/,
     .       '                        dVas = ',3(1x,g12.6))
      return
      end
c
      subroutine acalc95(r1,r2,r3,A,dA)
c---------------------------------------------------------------------
c  version of may 1, 1990
c  based on equations from notes date april 4, 1990
c  calculate the A value and its derivatives
c           A  = dabs[ (r1-r2)*(r2-r3)*(r3-r1) ]
c---------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      dimension dA(3)
      ipr = 0
      A = (r1-r2)*(r2-r3)*(r3-r1)
      dA(1) = ( -2.d0*r1 + r2 + r3 )*(r2-r3)
      dA(2) = ( -2.d0*r2 + r3 + r1 )*(r3-r1)
      dA(3) = ( -2.d0*r3 + r1 + r2 )*(r1-r2)
      if(A.lt.0.d0)then
         A = -A
         dA(1) = -dA(1)
         dA(2) = -dA(2)
         dA(3) = -dA(3)
      end if
      if(ipr.gt.0) write(7,7100) A,dA
 7100 format('Acalc enter ---------------------------------------',/,
     .       '     A = ',f15.10,'   dA = ',3(1x,g12.6),/,
     .       'Acalc exit ----------------------------------------')
      return
      end
c
      subroutine compac95(r,icompc,T,dT)
C---------------------------------------------------------------
c  version of may 14/90 ... calculates T and dT values also
c  decide whether or not this particular geometry is compact,
c  that is, are any of the three distances smaller than the
c  rr value from the johnson correction.
c-----------------------------------------------------------------
      implicit double precision (a-h,o-z)
      dimension r(3),T(3),dT(3)
      parameter( rr = 1.15d0, rp = 1.25d0 )
c  first see if this is a compact geometry:
      do i=1,3
	 T(i) = 0.d0
	 dT(i)= 0.d0
      end do
      icompc = 0
      if(r(1).lt.rr) icompc = icompc + 1
      if(r(2).lt.rr) icompc = icompc + 1
      if(r(3).lt.rr) icompc = icompc + 1
      if(icompc.eq.0) return
c calculate the T(i) values:
      do i=1,3
      if(r(i).lt.rr) then
	 top = rr-r(i)
	 bot = rp-r(i)
	 top2 = top  * top
	 top3 = top2 * top
	 bot2 = bot  * bot
         T(i)  = top3/bot
         dT(i) = -3.d0*top2/bot + top3/bot2
      end if
      end do
      return
      end
C
      subroutine casym95( r,Cas,dCas,id,T,dT )
c---------------------------------------------------------------
c apr12/95 ... surface950308 parameters added
c version of sept14/90
c  the compact asymmetric correction term and derivatives
c---------------------------------------------------------------
      implicit double precision (a-h,o-z)
      dimension r(3),T(3)
      dimension dPr(3),dT(3),dsumT(3),dterm1(3),determ(3),dseries(3),
     .          dCas(3),dA(3)
c Casym values jun21

      parameter(
     . u1=0.2210243144E+00, u2=0.4367417579E+00, u3=0.6994985432E-02,
     . u4=0.1491096501E+01, u5=0.1602896673E+01, u6=-.2821747323E+01,
     . u7=0.4948310833E+00, u8=-.3540394679E-01, u9=-.3305809954E+01,
     .u10=0.3644382172E+01,u11=-.9997570970E+00,u12=0.7989919534E-01,
     .u13=-.1075807322E-02) !jun21 fit
c---- Casym values mar08.95

c     parameter(
c    . u1=0.5120831287E+00, u2=0.1002277242E+01, u3=0.6850007419E-02,
c    . u4=-.2038751706E+01, u5=0.7027811909E+01, u6=-.4881767278E+01,
c    . u7=0.8801769106E+00, u8=-.6296419648E-01, u9=-.8125516783E+01,
c    .u10=0.6073964424E+01,u11=-.1451402523E+01,u12=0.1165084183E+00,
c    .u13=-.1176579871E-02)
c---- Casym values from fit633a
c     parameter(
c    . u1=0.1537481166E+00, u2=0.2745950036E+00, u3=0.7501206780E-02,
c    . u4=0.3119136023E+01, u5=0.9969170798E+00, u6=-.3373682823E+01,
c    . u7=0.6807215913E+00, u8=-.4920325491E-01, u9=-.3919467989E+01,
c    .u10=0.5085532326E+01,u11=-.1415264778E+01,u12=0.1138681785E+00,
c    .u13=-.1525367566E-02)
      ipr = 0
      cas = 0.d0
      dCas(1) = 0.d0
      dCas(2) = 0.d0
      dCas(3) = 0.d0
      call acalc95(r(1),r(2),r(3),A,dA)
      A2 = A*A
c     if(A.eq.0.d0) return
      sumT = T(1) + T(2) + T(3)
      Sr   = r(1) + r(2) + r(3)
      Pr   = r(1) * r(2) * r(3)
      Sr2  = Sr*Sr
      Sr3  = Sr2*Sr
      Pr2  = Pr*Pr
      Pr3  = Pr2*Pr
c    write out the series explicitly:
      series = 1.d0 + u4/Pr2 +  u5/Pr + u6 +   u7*Pr +  u8*Pr2
     .           + A*(u9/Pr2 + u10/Pr + u11 + u12*Pr + u13*Pr2)
      term1  = u1/Pr**u2
      eterm  = dexp(-u3*Sr3)
      Cas    = sumt * a2 * term1 * series * eterm
      if(ipr.gt.0) then
	    write(7,*)
            write(7,*) 'Casym: ------------------'
	    write(7,*) ' T(i) = ',T
	    write(7,*) 'dT(i) = ',dT
	    write(7,*) ' id = ',id
	    write(7,7400) series,term1,eterm,Cas
	    write(7,*) ' Cas = ',Cas
      end if
      if(id.gt.0)then
	    dPr(1) = r(2)*r(3)
	    dPr(2) = r(3)*r(1)
	    dPr(3) = r(1)*r(2)
	 do i=1,3
	    dsumt(i) = dt(i)
	    dterm1(i)=  -1.d0*u1*u2*Pr**(-u2-1.d0)*dPr(i)
	    determ(i)= eterm *(-3.d0*u3*Sr2)
	    dseries(i)=
     .        dPr(i)*( -2.d0*u4/Pr3  -u5/Pr2 +u7  +2.d0*u8*Pr    )
     .        +dA(i)*( u9/Pr2 +u10/Pr +u11 +u12*Pr +u13*Pr2    )
     .        +A*dPr(i)*( -2.d0*u9/Pr3 -u10/Pr2 +u12 + 2.d0*u13*Pr )
	    dCas(i) =
     .        dsumt(i)      * a2    * term1 * series * eterm
     .      + 2.d0*a*dA(i)  * sumt  * term1 * series * eterm
     .      + dterm1(i)     * sumt  * a2    * series * eterm
     .      + dseries(i)    * sumt  * a2    * term1  * eterm
     .      + determ(i)     * sumt  * a2    * term1  * series
         end do
	 if(ipr.gt.0) then
	    write(7,7500) dPr,dt,dterm1,determ,dseries,dCas
	    write(7,*) ' dCas = ',dCas
	 end if
      end if
      return
 7400 format('subr.Cas:   series = ',g12.6,'     term1 = ',g12.6,/,
     .       '             eterm = ',g12.6,'      Cas  = ',g12.6)
 7500 format('    derivatives: dPr = ',3(1x,g12.6),/,
     .       '                 dT  = ',3(1x,g12.6),/,
     .       '              dterm1 = ',3(1x,g12.6),/,
     .       '              determ = ',3(1x,g12.6),/,
     .       '             dseries = ',3(1x,g12.6),/,
     .       '              dCas   = ',3(1x,g12.6))
      end
c
      subroutine csym95(r,Cal,dCal)
c---------------------------------------------------------------
c version of apr12/95  950308 values
c version of oct12/90  surf636 values
c  calculate the 'compact all' correction term and derivatives
c  a correction term (added sept 11/89), which adds a small
c  correction to all compact geometries
c---------------------------------------------------------------
      implicit double precision (a-h,o-z)
      dimension r(3),dCal(3),G(3)
c     dimension t(3)
      dimension dG(3),sumv(3)
      parameter( rr=1.15d0, rp=1.25d0 )
c Csym values jun21

      parameter(
     . v1=-.2071708868E+00, v2=-.5672350377E+00, v3=0.9058780367E-02)
!jun21 fit
c---- Csym values mar08/95

c     parameter(
c    . v1=-.2210049400E+00, v2=-.7054469608E+00, v3=0.4088898394E-02)
c---- Csym values from fit633a

c     parameter(
c    . v1=-.2070776049E+00, v2=-.5350898737E+00, v3=0.1011861942E-01)
      cal   = 0.d0
      ipr   = 0
      Sr    = r(1)+r(2)+r(3)
      Sr2   = Sr*Sr
      Sr3   = Sr*Sr2
      exp3  = dexp( -v3*Sr3 )
      dexp3 = -3.d0*v3*Sr2*exp3
      do i=1,3
         ri      = r(i)
	 rrri    = rr-ri
	 rrri2   = rrri*rrri
	 rrri3   = rrri*rrri2
	 rpri    = rp-ri
	 rpri2   = rpri*rpri
         G(i)    = 0.d0
	 dG(i)   = 0.d0
	 sumv(i) = v1+v1*v2*ri
         if(ri.lt.rr) then
	    G(i)  = (rrri3/rpri) *sumv(i)
  	    dG(i) = (rrri3/rpri2)*sumv(i)
     .                -3.d0*(rrri2/rpri)*sumv(i)
     .                + (rrri3/rpri)*v1*v2
	 end if
      end do
      sumG = G(1) + G(2) + G(3)
      cal  = sumG*exp3
      dCal(1) = dG(1)*exp3 + sumG*dexp3
      dCal(2) = dG(2)*exp3 + sumG*dexp3
      dCal(3) = dG(3)*exp3 + sumG*dexp3
      if(ipr.gt.0)then
	 write(7,*)
         write(7,7100) Cal
	 write(7,7000) G,dG,sumv,dCal
	 write(7,*) 'exiting subr.csym'
      end if
 7100 format('Csym enter ----------------------------------------',/,
     .       '    Cal = ',1x,g20.14)
 7000 format('      G = ',3(1x,g12.6),/,
     .       '     dG = ',3(1x,g12.6),/,
     .       '   sumv = ',3(1x,g12.6),/,
     .       '   dCal = ',3(1x,g16.10),/,
     .       'Csym exit -----------------------------------------')
      return
      end
c
      subroutine vbcb95(rpass,icompc,T,dT,ipr,
     .                  Vbnda,Vbndb,dVbnda,dVbndb,
     .                  Cbnda,Cbndb,dCbnda,dCbndb)
C---------------------------------------------------------------
c apr12/95 surface950308 values added
c jul24/91 fit705 cbend values added
c feb27/91 modified to match equation in h3 paper more closely
c nov 4/90 Vbend coefficients now a,g  Cbend coeff's still c,d
c in this version, the derivatives are always calculated
c subroutines vbend, cbend and B1ab all combined into this one
c module in order to improve efficiency by not having to pass
c around B1a,B1b,B2,B3a,B3b functions and derivatives
c  B1a = 1 - sum of [ P1(cos(theta(i))) ]
c  B1b = 1 - sum of [ P3(cos(theta(i))) ]
c--------------------------------------------------------------
      implicit double precision (a-h,o-z)
c     arrays required for B1,B2,B3 calculations:
      dimension rpass(3),dB1a(3),dB1b(3),dB2(3),dB3(6)
c     dimension th(3)
c     arrays required for Vbend calculations:
      dimension dVbnda(3),dVbndb(3)
      dimension dVb1(3),dvb2(3),dvb3(3),dvb4(3),dvb5(3)
c     dimension dVbnd(3)
      dimension Vbnd(2)
      dimension vb(2,25)
c     arrays required for Cbend calculations:
      dimension Cbnds(2),dCbnds(2,3)
      dimension dCb1(3),dCb2(3),dCb3(3),dCb4(3),dCb5(3),dCb6(3),
     .          dCb7(3),dCb8(3)
      dimension T(3),dP(3)
      dimension dT(3),dCbnda(3),dCbndb(3)
      dimension cb(2,25)
      parameter( z58=0.625d0, z38=0.375d0 )
c     parameters required for Vbend calculations:
      parameter(beta1=0.52d0, beta2=0.052d0, beta3=0.79d0 )
c vbenda terms jun21         expterms= 0.52d0, 0.052d0, 0.79d0

      parameter(
     .a11=-.1838073394E+03,a12=0.1334593242E+02,a13=-.2358129537E+00,
     .a21=-.4668193478E+01,a22=0.7197506670E+01,a23=0.2162004275E+02,
     .a24=0.2106294028E+02,a31=0.4242962586E+01,a32=0.4453505045E+01,
     .a41=-.1456918088E+00,a42=-.1692657366E-01,a43=0.1279520698E+01,
     .a44=-.4898940075E+00,a51=0.1742295219E+03,a52=0.3142175348E+02,
     .a53=0.5152903406E+01) !jun21 fit
c vbendb terms jun21

      parameter(
     .g11=-.4765732725E+02,g12=0.3648933563E+01,g13=-.7141145244E-01,
     .g21=0.1002349176E-01,g22=0.9989856329E-02,g23=-.4161953634E-02,
     .g24=0.9075807910E-03,g31=-.2693628729E+00,g32=-.1399065763E-01,
     .g41=-.1417634346E-01,g42=-.4870024792E-03,g43=0.1312231847E+00,
     .g44=-.4409850519E-01,g51=0.5382970863E+02,g52=0.4587102824E+01,
     .g53=0.1768550515E+01) !jun21 fit
c    vbenda mar08/95   expterms= 0.52d0, 0.052d0, 0.79d0
c     parameter(
c    .a11=-.1779469255E+03,a12=0.1292511538E+02,a13=-.2284450520E+00,
c    .a21=-.4671094689E+01,a22=0.7810423901E+01,a23=0.2359968959E+02,
c    .a24=0.2293876132E+02,a31=0.4191795902E+01,a32=0.4456025797E+01,
c    .a41=-.1419862958E+00,a42=-.1667760578E-01,a43=0.1246997125E+01,
c    .a44=-.4777517290E+00,a51=0.1681125268E+03,a52=0.3067567046E+02,
c    .a53=0.4942041527E+01)
c    vbendb mar08/95
c     parameter(
c    .g11=-.4677295200E+02,g12=0.3587873970E+01,g13=-.7051720560E-01,
c    .g21=0.9549632701E-02,g22=0.9971623818E-02,g23=-.4040392363E-02,
c    .g24=0.9045746595E-03,g31=-.2746842691E+00,g32=-.1412945503E-01,
c    .g41=-.1337907240E-01,g42=-.6200741002E-03,g43=0.1257251291E+00,
c    .g44=-.4195584422E-01,g51=0.5287171694E+02,g52=0.4490366264E+01,
c    .g53=0.1735452866E+01)
c
c cbenda terms jun21/95

      parameter(
     .c11=0.1860299931E+04,c12=-.6134458037E+03,c13=0.7337207161E+02,
     .c14=-.2676717625E+04,c15=0.1344099415E+04,c21=0.1538913137E+03,
     .c22=0.4348007369E+02,c23=0.1719720677E+03,c24=0.2115963042E+03,
     .c31=-.7026089414E+02,c32=-.1300938992E+03,c41=0.1310273564E+01,
     .c42=-.6175149574E+00,c43=-.2679089358E+02,c44=0.5577477171E+01,
     .c51=-.3543353539E+04,c52=-.3740709591E+03,c53=0.7979303144E+02,
     .c61=-.1104230585E+04,c62=0.4603572025E+04,c63=-.5593496634E+04,
     .c71=-.1069406434E+02,c72=0.1021807153E+01,c73=0.6669828341E-01,
     .c74=0.4168542348E+02,c75=0.1751608567E+02,c81=0.9486883238E+02,
     .c82=-.1519334221E+02,c83=0.4024697252E+04,c84=-.2225159395E+02) !jun21
c           (last 4 parameters renumbered)
c cbendb terms jun21/95

      parameter(
     .d11=0.4203543357E+03,d12=-.4922474096E+02,d13=0.3362942544E+00,
     .d14=-.3827423082E+03,d15=0.1746726001E+03,d21=0.1699995737E-01,
     .d22=0.1513036778E-01,d23=0.2659119354E-01,d24=-.5760387483E-02,
     .d31=0.1020622621E+02,d32=0.1050536271E-01,d41=0.6836172780E+00,
     .d42=-.1627858240E+00,d43=-.6925485045E+01,d44=0.1632567385E+01,
     .d51=0.1083595009E+04,d52=0.4641431791E+01,d53=-.8233144461E+00,
     .d61=-.6157225942E+02,d62=0.3094361471E+03,d63=-.3299631143E+03,
     .d71=0.8866227120E+01,d72=-.1382126854E+01,d73=0.7620770145E-01,
     .d74=-.5145757859E+02,d75=0.2046097265E+01,d81=0.2540775558E+01,
     .d82=-.4889246569E+00,d83=-.1127439280E+04,d84=-.2269932295E+01) !jun21
c           (last 4 parameters renumbered)
c
c    cbenda mar08/95
c     parameter(
c    .c11=0.1983833377E+04,c12=-.7161674985E+03,c13=0.9480354622E+02,
c    .c14=-.2860199829E+04,c15=0.1424701828E+04,c21=0.1544592401E+03,
c    .c22=0.3684848293E+02,c23=0.1717816399E+03,c24=0.2060270139E+03,
c    .c31=-.1215015279E+03,c32=-.1370574974E+03,c41=0.8903858200E+00,
c    .c42=-.6496543267E+00,c43=-.2367464822E+02,c44=0.5098769966E+01,
c    .c51=0.6728307258E+03,c52=-.3825078986E+03,c53=0.8128185587E+02,
c    .c61=-.1141379097E+04,c62=0.4771127549E+04,c63=-.5713570141E+04,
c    .c71=-.7394045711E+01,c72=0.9888694698E+00,c73=0.7493009017E-01,
c    .c74=0.9590644736E+02,c75=0.1925781563E+02,c81=0.1001295468E+03,
c    .c82=-.1533150643E+02,c83=-.1850297943E+03,c84=-.2420454635E+02)
c     (last 4 parameters renumbered)
c    cbendb terms mar08/95
c     parameter(
c    .d11=0.4441478933E+03,d12=-.5345816836E+02,d13=0.1443665553E+01,
c    .d14=-.4155134339E+03,d15=0.1912235377E+03,d21=-.3697316400E-02,
c    .d22=0.1688497284E-01,d23=0.2330055461E-01,d24=-.4670024282E-02,
c    .d31=0.6492099426E+01,d32=0.1597054504E-01,d41=0.6701003237E+00,
c    .d42=-.1868909419E+00,d43=-.6709596498E+01,d44=0.1713120242E+01,
c    .d51=-.5022064388E+04,d52=0.2322997491E+01,d53=-.2710031511E+00,
c    .d61=-.7265968242E+02,d62=0.3593588680E+03,d63=-.3933599161E+03,
c    .d71=0.9312252325E+01,d72=-.1380396384E+01,d73=0.7373026488E-01,
c    .d74=-.5070954743E+02,d75=0.7931122062E+00,d81=0.3599502276E+01,
c    .d82=-.5609868931E+00,d83=0.4980220561E+04,d84=-.1087179983E+01)
c     (last 4 parameters renumbered)
c
c Vbenda values from fit634b (used for surf706)
c     parameter(
c    .a11=-.2918252280E+03,a12=0.2164569141E+02,a13=-.4005497543E+00,
c    .a21=-.2890774947E+01,a22=0.1032032542E+02,a23=0.2681748056E+02,
c    .a24=0.2633751055E+02,a31=0.6180641351E+01,a32=0.5037667041E+01,
c    .a41=-.1125570079E+00,a42=-.3176529304E-02,a43=0.9068915355E+00,
c    .a44=-.7228418516E+00,a51=0.2785898232E+03,a52=0.4764442446E+02,
c    .a53=0.8621545662E+01)
c   Vbendb values from fit634b (used for surf706)
c     parameter(
c    .g11=-.4241912309E+02,g12=0.2951365281E+01,g13=-.4840201514E-01,
c    .g21=-.1159168549E-03,g22=0.8688003567E-02,g23=-.3486923900E-02,
c    .g24=0.8312212839E-03,g31=0.5621810473E-01,g32=-.9776930747E-02,
c    .g41=-.1178456251E-01,g42=0.3491086729E-02,g43=0.7430516993E-01,
c    .g44=-.9643636957E-01,g51=0.4735533782E+02,g52=0.3001038808E+01,
c    .g53=0.1896630453E+01)
c
c parameters required for Cbend calculations:
c
c-- Cbenda values from fit706d.out
c     parameter(
c    .c11=0.7107064647E+04,c12=-.3728421267E+04,c13=0.1757654624E+03,
c    .c14=-.9725998132E+04,c15=0.4665086074E+04,c21=-.4435165986E+02,
c    .c22=0.1604477309E+03,c23=0.5805142925E+03,c24=0.6892349445E+03,
c    .c31=0.6581730442E+03,c32=0.6078389739E+02,c41=0.2885182566E+01,
c    .c42=-.1728169916E+01,c43=-.1119535503E+03,c44=0.4052536250E+02,
c    .c51=0.2540505673E+03,c52=-.5762083627E+03,c53=0.1295901320E+03,
c    .c61=-.2131706075E+04,c62=0.9084452020E+04,c63=-.1138253963E+05,
c    .c71=-.3964298833E+02,c72=-.5019979693E+01,c73=0.2906541488E+00,
c    .c74=0.1212943686E+04,c75=0.4140463415E+02,c81=0.1752855549E+03,
c    .c82=-.2496320107E+02,c83=0.3765413052E+03,c84=-.5480488130E+02)
c-- Cbendb values from fit706d.out
c     parameter(
c    .d11=0.1917166552E+04,d12=-.6542563392E+03,d13=0.6793758367E+02,
c    .d14=-.1694968847E+04,d15=0.6866649703E+03,d21=-.2137567948E+00,
c    .d22=0.4975938228E-01,d23=0.9364998295E-01,d24=-.2444320779E-01,
c    .d31=-.2863126914E+02,d32=0.5443219625E-01,d41=0.9673956120E+00,
c    .d42=-.1160159706E+01,d43=-.2424199759E+02,d44=0.8569424490E+01,
c    .d51=-.6517635862E+04,d52=0.1518098147E+03,d53=-.2706514366E+02,
c    .d61=0.4308392956E+02,d62=-.1234851732E+03,d63=0.2320626055E+03,
c    .d71=0.1049541418E+02,d72=-.2424169341E+01,d73=0.1745646946E+00,
c    .d74=0.2603615561E+02,d75=0.1345799970E+02,d81=-.6653710513E+01,
c    .d82=-.2576854447E+00,d83=0.6172608425E+04,d84=-.1328142473E+02)
c end of parameters
c
c   feb27/91 new c51 = c51+c83;   new d51 = d51+d83
      cx1 = c51 + c83
      dx1 = d51 + d83
      r1 = rpass(1)
      r2 = rpass(2)
      r3 = rpass(3)
      t1 = r1*r1 -r2*r2 -r3*r3
      t2 = r2*r2 -r3*r3 -r1*r1
      t3 = r3*r3 -r1*r1 -r2*r2
c  calculate the cosines of the three internal angles:
      c1 = t1 / (-2.d0*r2*r3)
      c2 = t2 / (-2.d0*r3*r1)
      c3 = t3 / (-2.d0*r1*r2)
      sum = c1 + c2 + c3
      B1a = 1.d0 - sum
      c1cube = c1*c1*c1
      c2cube = c2*c2*c2
      c3cube = c3*c3*c3
      cos3t1 = 4.d0*c1cube - 3.d0*c1
      cos3t2 = 4.d0*c2cube - 3.d0*c2
      cos3t3 = 4.d0*c3cube - 3.d0*c3
      sumb   = cos3t1 + cos3t2 + cos3t3
      B1b    = 1.d0 - ( z58*sumb + z38*sum )
c  calculate derivatives if desired
         dc1dr1 = -r1/(r2*r3)
         dc2dr2 = -r2/(r1*r3)
         dc3dr3 = -r3/(r1*r2)
         dc1dr2 = ( t1/(r2*r2) + 2.d0 )/(2.d0*r3)
         dc1dr3 = ( t1/(r3*r3) + 2.d0 )/(2.d0*r2)
         dc2dr1 = ( t2/(r1*r1) + 2.d0 )/(2.d0*r3)
         dc2dr3 = ( t2/(r3*r3) + 2.d0 )/(2.d0*r1)
         dc3dr1 = ( t3/(r1*r1) + 2.d0 )/(2.d0*r2)
         dc3dr2 = ( t3/(r2*r2) + 2.d0 )/(2.d0*r1)
         db1a(1) = -1.d0*( dc1dr1 + dc2dr1 + dc3dr1 )
         db1a(2) = -1.d0*( dc1dr2 + dc2dr2 + dc3dr2 )
         db1a(3) = -1.d0*( dc1dr3 + dc2dr3 + dc3dr3 )
         d1 = 12.d0*c1*c1 - 3.d0
         d2 = 12.d0*c2*c2 - 3.d0
         d3 = 12.d0*c3*c3 - 3.d0
         db1b(1) = -z58*( d1*dc1dr1 + d2*dc2dr1 + d3*dc3dr1 )
     .             -z38*(    dc1dr1 +    dc2dr1 +    dc3dr1 )
         db1b(2) = -z58*( d1*dc1dr2 + d2*dc2dr2 + d3*dc3dr2 )
     .             -z38*(    dc1dr2 +    dc2dr2 +    dc3dr2 )
         db1b(3) = -z58*( d1*dc1dr3 + d2*dc2dr3 + d3*dc3dr3 )
     .             -z38*(    dc1dr3 +    dc2dr3 +    dc3dr3 )
cd    if(ipr.gt.1)then
cd        write(7,*)
cd        write(7,*) ' from subr.vbcb -----------------------------'
cd        write(7,6000) rpass
cd        write(7,6010) b1a,b1b
cd        write(7,6020) db1a,db1b
cd        if(ipr.gt.2)then
cd           write(7,7000) dc1dr1,dc1dr2,dc1dr3,
cd   .                     dc2dr1,dc2dr2,dc2dr3,
cd   .                     dc3dr1,dc3dr2,dc3dr3
cd           write(7,7010) t1,t2,t3,c1,c2,c3,d1,d2,d3
cd        end if
cd    end if
c
c  calculate the quantities used by both Vbend and Cbend:
c
      R    = r1 + r2 + r3
      Rsq  = R*R
      B2   = 1.d0/r1 + 1.d0/r2 + 1.d0/r3
      B3   = (r2-r1)*(r2-r1) + (r3-r2)*(r3-r2) + (r1-r3)*(r1-r3)
      eps2 = 1.d-12
      B3b  = dsqrt( B3 + eps2 )
	 dB2(1) = -1.d0/(r1*r1)
	 dB2(2) = -1.d0/(r2*r2)
	 dB2(3) = -1.d0/(r3*r3)
	 dB3(1) = 4.d0*r1 -2.d0*r2 -2.d0*r3
	 dB3(2) = 4.d0*r2 -2.d0*r3 -2.d0*r1
	 dB3(3) = 4.d0*r3 -2.d0*r1 -2.d0*r2
	 dB3(4) = 0.5d0*dB3(1)/B3b
	 dB3(5) = 0.5d0*dB3(2)/B3b
	 dB3(6) = 0.5d0*dB3(3)/B3b
         exp1   = dexp(-beta1*R)
         exp2   = dexp(-beta2*Rsq)
         exp7   = dexp(-beta3*R)
	 dexp1  = -beta1*exp1
	 dexp2  = -2.d0*beta2*R*exp2
	 dexp7  = -beta3*exp7
c   do the vbnda calculations:
         B1    = B1a
         B12   = B1*B1
         B13   = B12*B1
         B14   = B13*B1
         B15   = B14*B1
         asum  = a11 +  a12*R +  a13*Rsq
         bsum  = a21*B12 +  a22*B13 +  a23*B14 +  a24*B15
	 csum  = a31*B1*exp1 +  a32*B12*exp2
         dsum1 = a41*exp1 +  a42*exp2
         dsum2 = a43*exp1 +  a44*exp2
         fsum  = a51 +  a52*R +  a53*Rsq
         vb(1,1) = B1*asum*exp1
         vb(1,2) = bsum * exp2
         vb(1,3) = B2*csum
         vb(1,4) = B1*B3*dsum1 + B1*B3b*dsum2
         vb(1,5) = B1* fsum  *exp7
         Vbnd(1) = vb(1,1)+vb(1,2)+vb(1,3)+vb(1,4)+vb(1,5)
c
	 dasum   = a12+2.d0*a13*R
	 dbsum   =  2.d0* a21*B1  + 3.d0* a22*B12
     .            + 4.d0* a23*B13 + 5.d0* a24*B14
	 ddsum1  =  a41*dexp1 +  a42*dexp2
	 ddsum2  =  a43*dexp1 +  a44*dexp2
	 dfsum   =  a52 +2.d0*a53*R
         dVb1(1) = dB1a(1)*asum*exp1 + B1*dasum*exp1 +B1*asum*dexp1
         dVb1(2) = dB1a(2)*asum*exp1 + B1*dasum*exp1 +B1*asum*dexp1
         dVb1(3) = dB1a(3)*asum*exp1 + B1*dasum*exp1 +B1*asum*dexp1
c
         dVb2(1) = dbsum*dB1a(1)*exp2 + bsum*dexp2
         dVb2(2) = dbsum*dB1a(2)*exp2 + bsum*dexp2
         dVb2(3) = dbsum*dB1a(3)*exp2 + bsum*dexp2
c
c calculate the Vb3 derivatives:
         dVb3(1)=dB2(1)*csum + B2*(  a31*dB1a(1)*exp1 +  a31*B1*dexp1
     .                  +2.d0* a32*B1*dB1a(1)*exp2 +  a32*B12*dexp2)
         dVb3(2)=dB2(2)*csum + B2*(  a31*dB1a(2)*exp1 +  a31*B1*dexp1
     .                  +2.d0* a32*B1*dB1a(2)*exp2 +  a32*B12*dexp2)
         dVb3(3)=dB2(3)*csum + B2*(  a31*dB1a(3)*exp1 +  a31*B1*dexp1
     .                  +2.d0* a32*B1*dB1a(3)*exp2 +  a32*B12*dexp2)
c
c calculate the Vb4 derivatives (may 27/90):
         dVb4(1)= dB1a(1)*B3 *dsum1+  B1*dB3(1)*dsum1 + B1*B3 *ddsum1
     .          + dB1a(1)*B3b*dsum2 + B1*dB3(4)*dsum2 + B1*B3b*ddsum2
         dVb4(2)= dB1a(2)*B3 *dsum1+  B1*dB3(2)*dsum1 + B1*B3 *ddsum1
     .          + dB1a(2)*B3b*dsum2 + B1*dB3(5)*dsum2 + B1*B3b*ddsum2
         dVb4(3)= dB1a(3)*B3 *dsum1+  B1*dB3(3)*dsum1 + B1*B3 *ddsum1
     .          + dB1a(3)*B3b*dsum2 + B1*dB3(6)*dsum2 + B1*B3b*ddsum2
c
c calculate the Vb5 derivatives:
	 dVb5(1) = dB1a(1)*fsum*exp7 + B1*dfsum*exp7 + B1*fsum*dexp7
	 dVb5(2) = dB1a(2)*fsum*exp7 + B1*dfsum*exp7 + B1*fsum*dexp7
	 dVb5(3) = dB1a(3)*fsum*exp7 + B1*dfsum*exp7 + B1*fsum*dexp7
c
c calculate the overall derivatives:
         dVbnda(1) = dvb1(1)+dvb2(1)+dvb3(1)+dvb4(1)+dvb5(1)
         dVbnda(2) = dvb1(2)+dvb2(2)+dvb3(2)+dvb4(2)+dvb5(2)
         dVbnda(3) = dvb1(3)+dvb2(3)+dvb3(3)+dvb4(3)+dvb5(3)
	 Vbnda = Vbnd(1)
c
c---- now do the vbendb calculations --------
         B1    = B1b
         B12   = B1*B1
         B13   = B12*B1
         B14   = B13*B1
         B15   = B14*B1
         asum  = g11 +  g12*R +  g13*Rsq
         bsum  = g21*B12 +  g22*B13 +  g23*B14 +  g24*B15
	 csum  = g31*B1*exp1 +  g32*B12*exp2
         dsum1 = g41*exp1 +  g42*exp2
         dsum2 = g43*exp1 +  g44*exp2
         fsum  = g51 +  g52*R +  g53*Rsq
         vb(2,1) = B1*asum*exp1
         vb(2,2) = bsum * exp2
         vb(2,3) = B2*csum
         vb(2,4) = B1*B3*dsum1 + B1*B3b*dsum2
         vb(2,5) = B1* fsum  *exp7
         Vbnd(2) = vb(2,1)+vb(2,2)+vb(2,3)+vb(2,4)+vb(2,5)
c
	 dasum   = g12+2.d0*g13*R
	 dbsum   =  2.d0* g21*B1  + 3.d0* g22*B12
     .             +4.d0* g23*B13 + 5.d0* g24*B14
	 ddsum1  =  g41*dexp1 +  g42*dexp2
	 ddsum2  =  g43*dexp1 +  g44*dexp2
	 dfsum   =  g52 +2.d0*g53*R
         dVb1(1) = dB1b(1)*asum*exp1 + B1*dasum*exp1 +B1*asum*dexp1
         dVb1(2) = dB1b(2)*asum*exp1 + B1*dasum*exp1 +B1*asum*dexp1
         dVb1(3) = dB1b(3)*asum*exp1 + B1*dasum*exp1 +B1*asum*dexp1
c
         dVb2(1) = dbsum*dB1b(1)*exp2 + bsum*dexp2
         dVb2(2) = dbsum*dB1b(2)*exp2 + bsum*dexp2
         dVb2(3) = dbsum*dB1b(3)*exp2 + bsum*dexp2
c
c calculate the Vb3 derivatives:
         dVb3(1)=dB2(1)*csum + B2*(  g31*dB1b(1)*exp1 +  g31*B1*dexp1
     .                  +2.d0* g32*B1*dB1b(1)*exp2 +  g32*B12*dexp2)
         dVb3(2)=dB2(2)*csum + B2*(  g31*dB1b(2)*exp1 +  g31*B1*dexp1
     .                  +2.d0* g32*B1*dB1b(2)*exp2 +  g32*B12*dexp2)
         dVb3(3)=dB2(3)*csum + B2*(  g31*dB1b(3)*exp1 +  g31*B1*dexp1
     .                  +2.d0* g32*B1*dB1b(3)*exp2 +  g32*B12*dexp2)
c
c calculate the Vb4 derivatives (may 27/90):
         dVb4(1)= dB1b(1)*B3 *dsum1+  B1*dB3(1)*dsum1 + B1*B3 *ddsum1
     .          + dB1b(1)*B3b*dsum2 + B1*dB3(4)*dsum2 + B1*B3b*ddsum2
         dVb4(2)= dB1b(2)*B3 *dsum1+  B1*dB3(2)*dsum1 + B1*B3 *ddsum1
     .          + dB1b(2)*B3b*dsum2 + B1*dB3(5)*dsum2 + B1*B3b*ddsum2
         dVb4(3)= dB1b(3)*B3 *dsum1+  B1*dB3(3)*dsum1 + B1*B3 *ddsum1
     .          + dB1b(3)*B3b*dsum2 + B1*dB3(6)*dsum2 + B1*B3b*ddsum2
c
c calculate the Vb5 derivatives:
	 dVb5(1) = dB1b(1)*fsum*exp7 + B1*dfsum*exp7 + B1*fsum*dexp7
	 dVb5(2) = dB1b(2)*fsum*exp7 + B1*dfsum*exp7 + B1*fsum*dexp7
	 dVb5(3) = dB1b(3)*fsum*exp7 + B1*dfsum*exp7 + B1*fsum*dexp7
c
c calculate the overall derivatives:
         dVbndb(1) = dvb1(1)+dvb2(1)+dvb3(1)+dvb4(1)+dvb5(1)
         dVbndb(2) = dvb1(2)+dvb2(2)+dvb3(2)+dvb4(2)+dvb5(2)
         dVbndb(3) = dvb1(3)+dvb2(3)+dvb3(3)+dvb4(3)+dvb5(3)
	 Vbndb = Vbnd(2)
c
c  now calculate the Cbend correction terms:
c  Cbnda uses the B1a formula and Cbndb uses the B1b formula
      if(icompc.eq.0) return
      sumT  = T(1) + T(2) + T(3)
      Rcu   = Rsq*R
      P     = r1*r2*r3
      Psq   = P*P
      Pcu   = Psq*P
      dP(1) = r2*r3
      dP(2) = r3*r1
      dP(3) = r1*r2
      Cbnds(1)  = 0.d0
      Cbnds(2)  = 0.d0
      dCbnda(1) = 0.d0
      dCbnda(2) = 0.d0
      dCbnda(3) = 0.d0
      dCbndb(1) = 0.d0
      dCbndb(2) = 0.d0
      dCbndb(3) = 0.d0
c  calculate exponentials and derivatives (common to cbnda and cbndb):
            exp7  = dexp( -beta2*Rcu )
	    dexp7 = -3.d0*beta2*Rsq*exp7
c ----- calculate the Cbnda correction term ----------------
       B1    = B1a
       B12   = B1*B1
       B13   = B12*B1
       B14   = B13*B1
       B15   = B14*B1
       asum  = c11 + c12*R + c13*Rsq + c14/R + c15/Rsq
       bsum  = c21*B12 + c22*B13 + c23*B14 + c24*B15
       csum  = c31*B1 *exp1  + c32*B12*exp2
       dsum1 = c41*exp1 + c42*exp2
       dsum2 = c43*exp1 + c44*exp2
       fsum  = cx1 + c52*R + c53*Rsq
       gsum  = c61 + c62/R + c63/Rsq
       aasum = c71 + c72*P + c73*Psq + c74/P + c75/Psq
       ffsum =       c81*P + c82*Psq + c84/Psq
       dasum = c12 + 2.d0*c13*R -c14/Rsq -2.d0*c15/Rcu
       dbsum = 2.d0*c21*B1 +3.d0*c22*B12 +4.d0*c23*B13 +5.d0*c24*B14
       ddsum1= c41*dexp1 + c42*dexp2
       ddsum2= c43*dexp1 + c44*dexp2
       dfsum = c52 + 2.d0*c53*R
       dgsum = -c62/Rsq - 2.d0*c63/Rcu
       daasum = c72 +2.d0*c73*P -c74/Psq -2.d0*c75/Pcu
       dffsum = c81 + 2.d0*c82*P -2.d0*c84/Pcu
         cb(1,1)  = B1*asum * exp1  /P
         cb(1,2)  = bsum * exp2
         cb(1,3)  = B2*csum
         cb(1,4)  = B1*B3*dsum1 + B1*B3b*dsum2
         cb(1,5)  = B1 * fsum * exp7 /P
         cb(1,6)  = B1 * gsum * exp7
         cb(1,7)  = B1 * aasum * exp2
         cb(1,8)  = B1 * ffsum * exp7
         Cbnds(1) = cb(1,1)+cb(1,2)+cb(1,3)+cb(1,4)+cb(1,5)
     .                     +cb(1,6)+cb(1,7)+cb(1,8)
c       calculate the derivatives:
	 dCb1(1) = dB1a(1)*asum*exp1/P + B1*dasum*exp1/P
     .                +B1*asum*dexp1/P - B1*asum*exp1*dP(1)/Psq
	 dCb1(2) = dB1a(2)*asum*exp1/P + B1*dasum*exp1/P
     .                +B1*asum*dexp1/P - B1*asum*exp1*dP(2)/Psq
	 dCb1(3) = dB1a(3)*asum*exp1/P + B1*dasum*exp1/P
     .                +B1*asum*dexp1/P - B1*asum*exp1*dP(3)/Psq
c
	 dCb2(1) = dbsum*dB1a(1)*exp2 + bsum*dexp2
	 dCb2(2) = dbsum*dB1a(2)*exp2 + bsum*dexp2
	 dCb2(3) = dbsum*dB1a(3)*exp2 + bsum*dexp2
c
	 dCb3(1) = dB2(1)*csum + B2*( c31*dB1a(1)*exp1 +c31*B1*dexp1
     .            +c32 *2.d0*B1*dB1a(1)*exp2 + c32 *B12*dexp2 )
	 dCb3(2) = dB2(2)*csum + B2*( c31 *dB1a(2)*exp1 +c31 *B1*dexp1
     .            +c32 *2.d0*B1*dB1a(2)*exp2 + c32 *B12*dexp2 )
	 dCb3(3) = dB2(3)*csum + B2*( c31 *dB1a(3)*exp1 +c31 *B1*dexp1
     .            +c32 *2.d0*B1*dB1a(3)*exp2 + c32 *B12*dexp2 )
c          dCb4 equations may 27/90:
	 dCb4(1) = dB1a(1)*B3 *dsum1 + B1*dB3(1)*dsum1 + B1*B3 *ddsum1
     .           + dB1a(1)*B3b*dsum2 + B1*dB3(4)*dsum2 + B1*B3b*ddsum2
	 dCb4(2) = dB1a(2)*B3 *dsum1 + B1*dB3(2)*dsum1 + B1*B3 *ddsum1
     .           + dB1a(2)*B3b*dsum2 + B1*dB3(5)*dsum2 + B1*B3b*ddsum2
	 dCb4(3) = dB1a(3)*B3 *dsum1 + B1*dB3(3)*dsum1 + B1*B3 *ddsum1
     .           + dB1a(3)*B3b*dsum2 + B1*dB3(6)*dsum2 + B1*B3b*ddsum2
c          dCb5 equations may 27/90: (corrected jun01)
	 dCb5(1) = dB1a(1)*fsum*exp7/P + B1*dfsum*exp7/P
     .               + B1*fsum*dexp7/P - B1*fsum*exp7*dP(1)/Psq
	 dCb5(2) = dB1a(2)*fsum*exp7/P + B1*dfsum*exp7/P
     .               + B1*fsum*dexp7/P - B1*Fsum*exp7*dP(2)/Psq
	 dCb5(3) = dB1a(3)*fsum*exp7/P + B1*dfsum*exp7/P
     .               + B1*fsum*dexp7/P - B1*Fsum*exp7*dP(3)/Psq
c
	 dCb6(1) = dB1a(1)*gsum*exp7 + B1*dgsum*exp7 + B1*gsum*dexp7
	 dCb6(2) = dB1a(2)*gsum*exp7 + B1*dgsum*exp7 + B1*gsum*dexp7
	 dCb6(3) = dB1a(3)*gsum*exp7 + B1*dgsum*exp7 + B1*gsum*dexp7
c
	 dCb7(1) = dB1a(1)*aasum*exp2 + B1*daasum*dP(1)*exp2
     .                               + B1* aasum*dexp2
	 dCb7(2) = dB1a(2)*aasum*exp2 + B1*daasum*dP(2)*exp2
     .                               + B1* aasum*dexp2
	 dCb7(3) = dB1a(3)*aasum*exp2 + B1*daasum*dP(3)*exp2
     .                               + B1* aasum*dexp2
	 dCb8(1) = dB1a(1)*ffsum*exp7  +B1*dffsum*dP(1)*exp7
     .                                +B1* ffsum*dexp7
	 dCb8(2) = dB1a(2)*ffsum*exp7  +B1*dffsum*dP(2)*exp7
     .                                +B1* ffsum*dexp7
	 dCb8(3) = dB1a(3)*ffsum*exp7  +B1*dffsum*dP(3)*exp7
     .                                +B1* ffsum*dexp7
           dCbnds(1,1) = dCb1(1) +dCb2(1) +dCb3(1) +dCb4(1) +dCb5(1)
     .                  +dCb6(1) +dCb7(1) +dCb8(1)
           dCbnds(1,2) = dCb1(2) +dCb2(2) +dCb3(2) +dCb4(2) +dCb5(2)
     .                  +dCb6(2) +dCb7(2) +dCb8(2)
           dCbnds(1,3) = dCb1(3) +dCb2(3) +dCb3(3) +dCb4(3) +dCb5(3)
     .                  +dCb6(3) +dCb7(3) +dCb8(3)
c
c ----- calculate the Cbndb correction term ----------------
       B1    = B1b
       B12   = B1*B1
       B13   = B12*B1
       B14   = B13*B1
       B15   = B14*B1
       asum  = d11 + d12*R + d13*Rsq + d14/R + d15/Rsq
       bsum  = d21*B12 + d22*B13 + d23*B14 + d24*B15
       csum  = d31*B1 *exp1  + d32*B12*exp2
       dsum1 = d41*exp1 + d42*exp2
       dsum2 = d43*exp1 + d44*exp2
       fsum  = dx1 + d52*R + d53*Rsq
       gsum  = d61 + d62/R + d63/Rsq
       aasum = d71 + d72*P + d73*Psq + d74/P + d75/Psq
       ffsum =       d81*P + d82*Psq + d84/Psq
       dasum = d12 + 2.d0*d13*R -d14/Rsq -2.d0*d15/Rcu
       dbsum = 2.d0*d21*B1 +3.d0*d22*B12 +4.d0*d23*B13 +5.d0*d24*B14
       ddsum1= d41*dexp1 + d42*dexp2
       ddsum2= d43*dexp1 + d44*dexp2
       dfsum = d52 + 2.d0*d53*R
       dgsum = -d62/Rsq - 2.d0*d63/Rcu
       daasum = d72 +2.d0*d73*P -d74/Psq -2.d0*d75/Pcu
       dffsum = d81 + 2.d0*d82*P -2.d0*d84/Pcu
         cb(2,1)  = B1*asum * exp1  /P
         cb(2,2)  = bsum * exp2
         cb(2,3)  = B2*csum
         cb(2,4)  = B1*B3*dsum1 + B1*B3b*dsum2
         cb(2,5)  = B1 * fsum * exp7 /P
         cb(2,6)  = B1 * gsum * exp7
         cb(2,7)  = B1 * aasum * exp2
         cb(2,8)  = B1 * ffsum * exp7
         Cbnds(2) = cb(2,1)+cb(2,2)+cb(2,3)+cb(2,4)+cb(2,5)
     .                     +cb(2,6)+cb(2,7)+cb(2,8)
c       calculate the derivatives:
	 dCb1(1) = dB1b(1)*asum*exp1/P + B1*dasum*exp1/P
     .                +B1*asum*dexp1/P - B1*asum*exp1*dP(1)/Psq
	 dCb1(2) = dB1b(2)*asum*exp1/P + B1*dasum*exp1/P
     .                +B1*asum*dexp1/P - B1*asum*exp1*dP(2)/Psq
	 dCb1(3) = dB1b(3)*asum*exp1/P + B1*dasum*exp1/P
     .                +B1*asum*dexp1/P - B1*asum*exp1*dP(3)/Psq
c
	 dCb2(1) = dbsum*dB1b(1)*exp2 + bsum*dexp2
	 dCb2(2) = dbsum*dB1b(2)*exp2 + bsum*dexp2
	 dCb2(3) = dbsum*dB1b(3)*exp2 + bsum*dexp2
c
	 dCb3(1)= dB2(1)*csum + B2*( d31 *dB1b(1)*exp1 +d31 *B1*dexp1
     .           +d32 *2.d0*B1*dB1b(1)*exp2 + d32 *B12*dexp2 )
	 dCb3(2)= dB2(2)*csum + B2*( d31 *dB1b(2)*exp1 +d31 *B1*dexp1
     .           +d32 *2.d0*B1*dB1b(2)*exp2 + d32 *B12*dexp2 )
	 dCb3(3)= dB2(3)*csum + B2*( d31 *dB1b(3)*exp1 +d31 *B1*dexp1
     .           +d32 *2.d0*B1*dB1b(3)*exp2 + d32 *B12*dexp2 )
c          dCb4 equations may 27/90:
	 dCb4(1) = dB1b(1)*B3 *dsum1 + B1*dB3(1)*dsum1 + B1*B3 *ddsum1
     .           + dB1b(1)*B3b*dsum2 + B1*dB3(4)*dsum2 + B1*B3b*ddsum2
	 dCb4(2) = dB1b(2)*B3 *dsum1 + B1*dB3(2)*dsum1 + B1*B3 *ddsum1
     .           + dB1b(2)*B3b*dsum2 + B1*dB3(5)*dsum2 + B1*B3b*ddsum2
	 dCb4(3) = dB1b(3)*B3 *dsum1 + B1*dB3(3)*dsum1 + B1*B3 *ddsum1
     .           + dB1b(3)*B3b*dsum2 + B1*dB3(6)*dsum2 + B1*B3b*ddsum2
c          dCb5 equations may 27/90:
	 dCb5(1) = dB1b(1)*fsum*exp7/P + B1*dfsum*exp7/P
     .               + B1*fsum*dexp7/P - B1*fsum*exp7*dP(1)/Psq
	 dCb5(2) = dB1b(2)*fsum*exp7/P + B1*dfsum*exp7/P
     .               + B1*fsum*dexp7/P - B1*fsum*exp7*dP(2)/Psq
	 dCb5(3) = dB1b(3)*fsum*exp7/P + B1*dfsum*exp7/P
     .               + B1*fsum*dexp7/P - B1*fsum*exp7*dP(3)/Psq
c
	 dCb6(1) = dB1b(1)*gsum*exp7 + B1*dgsum*exp7 + B1*gsum*dexp7
	 dCb6(2) = dB1b(2)*gsum*exp7 + B1*dgsum*exp7 + B1*gsum*dexp7
	 dCb6(3) = dB1b(3)*gsum*exp7 + B1*dgsum*exp7 + B1*gsum*dexp7
c
	 dCb7(1) = dB1b(1)*aasum*exp2 + B1*daasum*dP(1)*exp2
     .                               + B1* aasum*dexp2
	 dCb7(2) = dB1b(2)*aasum*exp2 + B1*daasum*dP(2)*exp2
     .                               + B1* aasum*dexp2
	 dCb7(3) = dB1b(3)*aasum*exp2 + B1*daasum*dP(3)*exp2
     .                               + B1* aasum*dexp2
	 dCb8(1) = dB1b(1)*ffsum*exp7  +B1*dffsum*dP(1)*exp7
     .                                +B1* ffsum*dexp7
	 dCb8(2) = dB1b(2)*ffsum*exp7  +B1*dffsum*dP(2)*exp7
     .                                +B1* ffsum*dexp7
	 dCb8(3) = dB1b(3)*ffsum*exp7  +B1*dffsum*dP(3)*exp7
     .                                +B1* ffsum*dexp7
           dCbnds(2,1) = dCb1(1) +dCb2(1) +dCb3(1) +dCb4(1) +dCb5(1)
     .                  +dCb6(1) +dCb7(1) +dCb8(1)
           dCbnds(2,2) = dCb1(2) +dCb2(2) +dCb3(2) +dCb4(2) +dCb5(2)
     .                  +dCb6(2) +dCb7(2) +dCb8(2)
           dCbnds(2,3) = dCb1(3) +dCb2(3) +dCb3(3) +dCb4(3) +dCb5(3)
     .                  +dCb6(3) +dCb7(3) +dCb8(3)
c
c  calculate the total derivative from the pieces:
         dCbnda(1) = dT(1) * Cbnds(1)    + sumT * dCbnds(1,1)
         dCbndb(1) = dT(1) * Cbnds(2)    + sumT * dCbnds(2,1)
         dCbnda(2) = dT(2) * Cbnds(1)    + sumT * dCbnds(1,2)
         dCbndb(2) = dT(2) * Cbnds(2)    + sumT * dCbnds(2,2)
         dCbnda(3) = dT(3) * Cbnds(1)    + sumT * dCbnds(1,3)
         dCbndb(3) = dT(3) * Cbnds(2)    + sumT * dCbnds(2,3)
      Cbnda = sumT*Cbnds(1)
      Cbndb = sumT*Cbnds(2)
c  some debugging print statements:
cd    if(ipr.gt.1)then
cd       write(7,7650) B2,B3
cd       write(7,7700) db2,db3
cd       write(7,*)
cd       write(7,*) 'Vbend values :'
cd       k=1
cd       write(7,7150) k,dvb1,dvb2,dvb3,dvb4,dvb5,dVbnda
cd       write(7,7155) (vb(1,i),i=1,5)
cd       k=2
cd       write(7,7150) k,dvb1,dvb2,dvb3,dvb4,dvb5,dVbndb
cd       write(7,7155) (vb(2,i),i=1,5)
cd       write(7,7100) Vbnda,Vbndb
cd       write(7,*)
cd       write(7,*) 'Cbend values:'
cd       write(7,*) 'c81,c81,c82=',c81,c81,c82
cd       write(7,*) 'c91,c83,c84=',c91,c83,c84
cd       write(7,*) 'd81,d81,d82=',d81,d81,d82
cd       write(7,*) 'd91,d83,d84=',d91,d83,d84
cd       k=1
cd          write(7,2020) k,(cb(1,j),j=1,9)
cd          write(7,2033) sumT,Cbnds(1)
cd       k=2
cd          write(7,2020) k,(cb(k,j),j=1,9)
cd          write(7,2033) sumT,Cbnds(2)
cd       k=1
cd       write(7,7152) k,dcb1,dcb2,dcb3,dcb4,dcb5,dcb6,
cd   .               dcb7,dcb8,dcb9,(dCbnds(1,i),i=1,3)
cd       k=2
cd       write(7,7152) k,dcb1,dcb2,dcb3,dcb4,dcb5,dcb6,
cd   .                 dcb7,dcb8,dcb9,(dCbnds(k,i),i=1,3)
cd       write(7,7200) dT,dCbnda,dCbndb
cd       write(7,*) 'VbCb  exit -------------------------'
cd       write(7,*)
cd    end if
      return
 2000 format(5x,'r1, r2, r3 = ',3(1x,f12.8))
 2010 format(5x,'B1a, B1b, B2, B3 = ',4(1x,g12.6))
 7100 format('    Vbnda = ',g12.6,'     Vbndb = ',g12.6)
 7650 format('    B2 =    ',g12.6,'       B3 =  ',g12.6)
 7700 format('    dB2 =   ',3(1x,g12.6),/,
     .       '    dB3 =   ',6(1x,g12.6))
 7150 format(' k=',i1,'     derivatives: ',/,
     .       ' dVb1  = ',3(1x,g16.10),/,
     .       ' dVb2  = ',3(1x,g16.10),/,
     .       ' dVb3  = ',3(1x,g16.10),/,
     .       ' dVb4  = ',3(1x,g16.10),/,
     .       ' dVb5  = ',3(1x,g16.10),/,
     .       ' dVbnd = ',3(1x,g16.10))
 7155 format('  Vb1 = ',g16.10,'  Vb2 = ',g16.10,/,
     .       '  Vb3 = ',g16.10,'  Vb4 = ',g16.10,/,
     .       '  Vb5 = ',g16.10)
 2020 format(5x,'k=',i1,' cb1,cb2,cb3= ',3(1x,g16.10),
     .     /,8x,' cb4,cb5,cb6= ',3(1x,g16.10),
     .     /,8x,' cb7,cb8,cb9= ',3(1x,g16.10))
 2033 format(5x,'sumT = ',g12.6 ,'  cbnds(k) = ',g12.6)
 7152 format('Subr.Cbend:   k=',i1,'     derivatives: ',/,
     .       '  dCb1  = ',3(1x,g18.12),/,
     .       '  dCb2  = ',3(1x,g18.12),/,
     .       '  dCb3  = ',3(1x,g18.12),/,
     .       '  dCb4  = ',3(1x,g18.12),/,
     .       '  dCb5  = ',3(1x,g18.12),/,
     .       '  dCb6  = ',3(1x,g18.12),/,
     .       '  dCb7  = ',3(1x,g18.12),/,
     .       '  dCb8  = ',3(1x,g18.12),/,
     .       '  dCb9  = ',3(1x,g18.12),/,
     .       '  dCbnd = ',3(1x,g18.12))
 7200 format('  dT    = ',3(1x,g18.12),/,
     .       '  dCbnda= ',3(1x,g18.12),/,
     .       '  dCbndb= ',3(1x,g18.12))
 6000 format(' from subr.b1ab:  r=',3(1x,f9.6))
 6010 format('    b1a = ',f16.10,'     b1b = ',f16.10)
 6020 format('    db1a: ',3(1x,e12.6),/,'    db1b: ',3(1x,e12.6))
 7000 format('    dc(j)/dr(i) derivatives: ',/,
     .    8x,3(1x,e12.6),/,8x,3(1x,e12.6),/,8x,3(1x,e12.6))
 7010 format('    ri2-rj2-rk2: ',3(1x,e12.6),/,
     .       '       cosines : ',3(1x,e12.6),/,
     .       '    12c**2 - 3 : ',3(1x,e12.6))
      end
c
      subroutine chgeom(r,ivalid)
c----------------------------------------------------------------------c
c apr21/95
c first, check that it is a valid H3 geometry, within tolerance "derr":
c this test corrected on nov23/93
c     generally, rlo + rmed > rhi
c     sometimes, rlo + rmed = rhi (linear geometry)
c     but if     rlo + rmed < rhi there's a problem
c
c error1 ... check that rlo+rmid>rhi
c istop1 ... [0] warning [1]stop calculations
c
c error2 ... check that all r(i)'s are > 0.2 bohr
c istop2 ... [0] warning [1]stop calculations
c----------------------------------------------------------------------c
      implicit double precision (a-h,o-z)
      parameter(derr = 1.d-5)
      parameter( istop1 = 0 , istop2 = 0 )
      dimension r(3)
      ivalid = 1
      rhi = dmax1(r(1),r(2))
      if(r(3).ge.rhi)then
         rmid = rhi
         rhi = r(3)
      else
         rmid = dmax1( r(3) , dmin1(r(1),r(2)) )
      endif
      rlo = dmin1(r(1),r(2),r(3))
cx    if(rlo+rmid.gt.rhi+derr)then  {old condition}
      if(rlo+rmid+derr.lt.rhi)then
cd       write(6,*) ' rlongest          = ',rhi+derr
cd       write(6,*) ' rmiddle,rshortest = ',rmid,rlo
c        write(6,*) 'Warning: invalid geometry ',r
         if(istop1.gt.0) stop ' STOP -- geometry error '
c         ivalid = -1
      end if
      if(rlo.lt.0.2d0)then
cd       write(6,*) ' shortest distance: ',rlo
c        write(6,*) 'Warning: invalid geometry ',r
         if(istop2.gt.0) stop ' STOP -- geometry error '
c         ivalid = -2
      end if
      return
      end
c===================================================================
      subroutine fh2pes0(rin,v)
      implicit real*8(a-h,o-z)
c   r1=r(H-H), r2=r(F-H)  in A
   	parameter (rbohr=0.5291771d0,scale=0.996d0)
	parameter (pi=3.141592653589793d0)
        parameter (nhh=80,nfh=30,nth=19)
         common /pes/rhh(nth,nfh,nhh),rfh(nfh),thth(nth),vcut,
     & ve(nth,nfh,nhh),ind(nth,nfh)
	dimension rin(3)
        data init/0/
        save init

        if(init.eq.0) then
          call potread(2)
          init=1
c         read(288,*) fac
        end if

      open(98,file='fh2av5z-final.txt')

	ich=0
	r1=rin(1)
	r2=rin(2)
	r3=rin(3)
	cthi=(r1*r1+r2*r2-r3*r3)/(2.0d0*r1*r2)
	cth=cthi
        ctha=(r1*r1+r3*r3-r2*r2)/(2.0d0*r1*r3)
	if (r2.gt.r3) then
          r2=r3
	  cth=ctha
	endif
	if (cth.lt.-1.0d0) cth=-1.0d0
	if (cth.gt.1.0d0) cth=1.0d0

        th=dacos(cth)*180.0d0/pi
	if (r1.lt.0.7d0) then
           v=vcut
	   return
        endif
	if (r2.lt.1.2d0) then
           v=vcut
	   return
	endif
	if (r1.gt.13.d0) r1=13.d0
	if (r2.gt.11.d0) r2=11.d0
	CALL SPl3(r1,r2,th,va,1)
        r30=dsqrt(r10*r10+r20*r20-2.0d0*r10*r20*cthi)
c        v2=pot_FHH_HSW(rin(2),rin(1),rin(3),va)
	va=va*scale
c        v=va+v2
        v=va-0.000000944376d0
      end

c======================================================================
	
c   ab inito potential energy surface for the ground electronic state of FH2
c      Please add the following statement in your main program:
c           call potread(ipes)
c   ipes=1 for CBS
c   ipes=2 for av5z
c     use the following statement to calculate the PES when needed:
c                 call fh2pes(r1,r2,cth,v)
c      where r1 and r2 are the two bondlength hh and hf in au,
c            th is the enclosed angle in degree,cth=cos(th)
c       v is the potential in hartree
cmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmc
      double precision function pot_FHH_HSW(rhf1,r,rhf2)
c given one point in distance coordinates (bohr), this routine returns the
c value of the HSW potential (hartree), or, optionally (see comments below)
c the value of the SW1 potential (hartree).
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c The S.O.-corrected Hartke-Stark-Werner potential for FH2 will be published in
c   B. Hartke, K. Stark and H.-J. Werner, in preparation                      c
c The SW1 potential without S.O.-correction has been published in:            c
c   K. Stark and H.-J. Werner, J.Chem.Phys. 104 (1996) 6515.                  c
c For related work, see also:                                                 c
c   B. Hartke and H.-J. Werner, Chem.Phys.Lett. 280 (1997) 430.               c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit double precision(a-h,o-z)
      parameter (np=40)
      dimension x(np),y(np),z(np),en(np,np,np)
      dimension rr(3),fin(4)
      data amuau,evau,aukcal/1822.88734d0,27.211608d0,627.5095d0/
      data ifirst/-1/
      save ifirst,pi,raddeg,nx,ny,nz,x,y,z,en,esocut,rsolim,rrsolm

      !!--PESLIB--!!
      CHARACTER*256 PARAPATH
      INTEGER*4 PARAPATHLEN
c
c on first call of this subroutine, initialize the SW1 potential routine
c and the spline interpolation for the S.O. correction
c
      if (ifirst.eq.-1) then
c         call inifit_FHH_HSW
         call prepar_FHH_HSW(nx,ny,nz,x,y,z,en)

         open(unit=83,file='so.param',status='old')
         read(83,*)esocut
         read(83,*)rsolim
         read(83,*)rrsolm
         close(83)
         esocut=esocut/aukcal

         pi=acos(-1.0d0)
         raddeg=180.d0/pi
         ifirst=0
      end if

c call sw1 potential routine
      cth=(rhf1*rhf1+r*r-rhf2*rhf2)/(2.0d0*rhf1*r)
c      call fh2pes(r,rhf1,cth,v)
c       write(*,*)'vvvvv=',r,rhf1,v
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
        call naturl_FHH_HSW(rr,0,fin)
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
      pot_FHH_HSW=socorr+0.0006119d0
c in order to get the SW1 potential, just omit the S.O. correction and shift
c      pot=v
c in that case, you can also comment out all the above parts relating to
c computation of the "socorr" value (spline interpolation, etc.)
      return
      end
c------------------------------------------------------------------------
!Subject: sospline.f

      subroutine prepar_FHH_HSW(nx,ny,nz,x,y,z,en)
      PARAMETER (NXMAX=40,NYMAX=40,NZMAX=40)
      real*8 x(NXMAX),y(NYMAX),z(NZMAX),en(NXMAX,NYMAX,NZMAX)

      !!--PESLIB--!!
      CHARACTER*256 PARAPATH
      INTEGER*4 PARAPATHLEN
      !!
      open(unit=109,file='so.input')

      read(109,*) nx,ny,nz
      do 10 i=1,nx
         do 11 j=1,ny
            do 12 k=1,nz
               read(109,*) iii,x(i),y(j),z(k),en(i,j,k)
   12       continue
   11    continue
   10 continue
 
      CLOSE(109)
c
c.....generate spline fit
c
      call spmain_FHH_HSW(nx,ny,nz,x,y,z,en)
      return
      end
c------------------------------------------------------
      subroutine spmain_FHH_HSW(nax,nay,naz,xa,ya,za,ea)                
     $        
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)                               
      PARAMETER (NXMAX=40,NYMAX=40,NZMAX=40,NSMAX=NXMAX*NYMAX*NZMAX)
      real*8 xa(NXMAX),ya(NYMAX),za(NZMAX),ea(NXMAX,NYMAX,NZMAX)
      COMMON /SURFAC_FHH_HSW/ F(NSMAX),                                 
     $        
     & PX(NSMAX),PY(NSMAX),PZ(NSMAX),                                   
     & PXY(NSMAX),PXZ(NSMAX),PYZ(NSMAX),                                
     & PXYZ(NSMAX)                                                      
      COMMON /GRIDX_FHH_HSW/  X(NXMAX),NX                               
     $        
      COMMON /GRIDY_FHH_HSW/  Y(NYMAX),NY                               
     $        
      COMMON /GRIDZ_FHH_HSW/  Z(NZMAX),NZ                               
     $        
C                                                                      
      NR(I,J,K)=(K-1)*NX*NY+(J-1)*NX+I                                 
C                                                                      
      nx=nax
      ny=nay
      nz=naz
      if(nx.gt.nxmax) stop 'nxmax'
      if(ny.gt.nymax) stop 'nymax'
      if(nz.gt.nzmax) stop 'nzmax'
      do 70 k=1,nz
      z(k)=za(k)
      do 70 j=1,ny
      y(j)=ya(j)
      do 70 i=1,nx
      x(i)=xa(i)
70    F(NR(I,J,K))=ea(i,j,k)                                            
      CALL SPLINF_FHH_HSW(2)                                            
     $       
        return
      END                                                              
c                                                                      
c********************************************************************* 
      SUBROUTINE SPLINF_FHH_HSW(INX)                                    
     $       
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)                               
      PARAMETER (NXMAX=40,NYMAX=40,NZMAX=40,NSMAX=NXMAX*NYMAX*NZMAX)
      common/SPF_FHH_HSW/ HX(NXMAX),HY(NXMAX),HZ(NXMAX),RLX(NXMAX),RMUX(
     $NXMAX),
     & RLY(NXMAX),RMUY(NXMAX),RLZ(NXMAX),RMUZ(NXMAX),XI(NXMAX),B(NXMAX),
     & AB(4),YZ(4)     
      COMMON /SURFAC_FHH_HSW/ F(NSMAX),                                 
     $        
     & PX(NSMAX),PY(NSMAX),PZ(NSMAX),                                   
     & PXY(NSMAX),PXZ(NSMAX),PYZ(NSMAX),                                
     & PXYZ(NSMAX)                                                      
      COMMON /GRIDX_FHH_HSW/  X(NXMAX),NX                               
     $        
      COMMON /GRIDY_FHH_HSW/  Y(NYMAX),NY                               
     $        
      COMMON /GRIDZ_FHH_HSW/  Z(NZMAX),NZ                               
     $        
      dimension spff(11*nxmax+8)
      equivalence (spff(1),hx(1))
      DATA UN/1.0D0/,THREE/3.0D0/                                      
C                                                                      
      NR(I,J,K)=(K-1)*NX*NY+(J-1)*NX+I                                 
C                                                                      
c     open(20,file = "spline3.tmp",status ="unkown",form="unformatted")
      NX2=NX-2                                                         
      NY2=NY-2                                                         
      NZ2=NZ-2                                                         
c.....zero out commons
      do 1 i=1,nx*ny*nz
      px(i)=0
      py(i)=0
      pz(i)=0
      pxy(i)=0
      pxz(i)=0
      pyz(i)=0
      pxyz(i)=0
    1 continue
      do 2 i=1,11*nxmax+8
    2 spff(i)=0
C     CALCUL DES HX,HY & HZ                                            
      DO 10 I=2,NX                                                     
   10 HX(I-1)=X(I)-X(I-1)                                              
      DO 20 J=2,NY                                                     
   20 HY(J-1)=Y(J)-Y(J-1)                                              
      DO 30 K=2,NZ                                                     
   30 HZ(K-1)=Z(K)-Z(K-1)                                              
C                                                                      
C     CALCUL DES LAMBDA & MU                                           
      DO 40 I=1,NX2                                                    
      RLX(I)=HX(I+1)/(HX(I)+HX(I+1))                                   
   40 RMUX(I)=UN-RLX(I)                                                
      DO 50 J=1,NY2                                                    
      RLY(J)=HY(J+1)/(HY(J)+HY(J+1))                                   
   50 RMUY(J)=UN-RLY(J)                                                
      DO 60 K=1,NZ2                                                    
      RLZ(K)=HZ(K+1)/(HZ(K)+HZ(K+1))                                   
   60 RMUZ(K)=UN-RLZ(K)                                                
C                                                                      
C     SPLINE-FIT DE P(X)IJK                                            
      MAN=NX-3                                                         
      DO 100 J=1,NY                                                    
      DO 100 K=1,NZ                                                    
      DO 110 I=1,4                                                     
      AB(I)=F(NR(I,J,K))                                               
  110 YZ(I)=F(NR(NX+I-4,J,K))                                          
      P0=DLAGRA_FHH_HSW(X,AB,4,1)                                       
     $       
      PN=DLAGRA_FHH_HSW(X(MAN),YZ,4,4)                                  
     $       
      PX(NR(1,J,K))=P0                                                 
      PX(NR(NX,J,K))=PN                                                
C     CALCUL SECOND MEMBRE                                             
      DO 120 I=1,NX2                                                   
  120 B(I)=THREE*RLX(I)/HX(I)*(F(NR(I+1,J,K))-F(NR(I,J,K)))            
     & +THREE*RMUX(I)/HX(I+1)*(F(NR(I+2,J,K))-F(NR(I+1,J,K)))          
      B(1)=B(1)-RLX(1)*P0                                              
      B(NX2)=B(NX2)-RMUX(NX2)*PN                                       
      CALL JORDAN_FHH_HSW(RMUX,RLX,XI,NX2,B)                            
     $       
      DO 100 I=1,NX2                                                   
  100 PX(NR(I+1,J,K))=XI(I)                                            
C                                                                      
C     SPLINE-FIT DE P(Y)                                               
      MAN=NY-3                                                         
      DO 200 I=1,NX                                                    
      DO 200 K=1,NZ                                                    
      DO 210 J=1,4                                                     
      AB(J)=F(NR(I,J,K))                                               
  210 YZ(J)=F(NR(I,NY+J-4,K))                                          
      P0=DLAGRA_FHH_HSW(Y,AB,4,1)                                       
     $       
      PN=DLAGRA_FHH_HSW(Y(MAN),YZ,4,4)                                  
     $       
      PY(NR(I,1,K))=P0                                                 
      PY(NR(I,NY,K))=PN                                                
C                                                                      
      DO 220 J=1,NY2                                                   
  220 B(J)=THREE*RLY(J)/HY(J)*(F(NR(I,J+1,K))-F(NR(I,J,K)))            
     & +THREE*RMUY(J)/HY(J+1)*(F(NR(I,J+2,K))-F(NR(I,J+1,K)))          
      B(1)=B(1)-RLY(1)*P0                                              
      B(NY2)=B(NY2)-RMUY(NY2)*PN                                       
      CALL JORDAN_FHH_HSW(RMUY,RLY,XI,NY2,B)                            
     $       
      DO 200 J=1,NY2                                                   
  200 PY(NR(I,J+1,K))=XI(J)                                            
C                                                                      
      if(nz.ne.1)then
C     SPLINE-FIT DE P(Z)                                               
      MAN=NZ-3                                                         
      DO 300 I=1,NX                                                    
      DO 300 J=1,NY                                                    
      DO 310 K=1,4                                                     
      AB(K)=F(NR(I,J,K))                                               
  310 YZ(K)=F(NR(I,J,NZ+K-4))                                          
      P0=DLAGRA_FHH_HSW(Z,AB,4,1)                                       
     $       
      PN=DLAGRA_FHH_HSW(Z(MAN),YZ,4,4)                                  
     $       
      PZ(NR(I,J,1))=P0                                                 
      PZ(NR(I,J,NZ))=PN                                                
C                                                                      
      DO 320 K=1,NZ2                                                   
  320 B(K)=THREE*RLZ(K)/HZ(K)*(F(NR(I,J,K+1))-F(NR(I,J,K)))            
     & +THREE*RMUZ(K)/HZ(K+1)*(F(NR(I,J,K+2))-F(NR(I,J,K+1)))          
      B(1)=B(1)-RLZ(1)*P0                                              
      B(NZ2)=B(NZ2)-RMUZ(NZ2)*PN                                       
      CALL JORDAN_FHH_HSW(RMUZ,RLZ,XI,NZ2,B)                            
     $       
      DO 300 K=1,NZ2                                                   
  300 PZ(NR(I,J,K+1))=XI(K)                                            
      endif
C                                                                      
C     SPLINE-FIT DE P(X,Y)                                             
      MAN=NY-3                                                         
      DO 400 I=1,NX                                                    
      DO 400 K=1,NZ                                                    
      GOTO (430,440),INX                                               
  430 DO 410 J=1,4                                                     
      AB(J)=PX(NR(I,J,K))                                              
  410 YZ(J)=PX(NR(I,NY+J-4,K))                                         
      P0=DLAGRA_FHH_HSW(Y,AB,4,1)                                       
     $       
      PN=DLAGRA_FHH_HSW(Y(MAN),YZ,4,4)                                  
     $       
      GOTO 450                                                         
  440 P0=0.                                                            
      PN=0.                                                            
  450 PXY(NR(I,1,K))=P0                                                
      PXY(NR(I,NY,K))=PN                                               
C                                                                      
      DO 420 J=1,NY2                                                   
  420 B(J)=THREE*RLY(J)/HY(J)*(PX(NR(I,J+1,K))-PX(NR(I,J,K)))          
     & +THREE*RMUY(J)/HY(J+1)*(PX(NR(I,J+2,K))-PX(NR(I,J+1,K)))        
      B(1)=B(1)-RLY(1)*P0                                              
      B(NY2)=B(NY2)-RMUY(NY2)*PN                                       
      CALL JORDAN_FHH_HSW(RMUY,RLY,XI,NY2,B)                            
     $       
      DO 400 J=1,NY2                                                   
  400 PXY(NR(I,J+1,K))=XI(J)                                           
C                                                                      
C     SPLINE-FIT DE P(X,Z)                                             
      if(nz.ne.1)then
      MAN=NZ-3                                                         
      DO 500 I=1,NX                                                    
      DO 500 J=1,NY                                                    
      GOTO (530,540),INX                                               
  530 DO 510 K=1,4                                                     
      AB(K)=PX(NR(I,J,K))                                              
  510 YZ(K)=PX(NR(I,J,K+NZ-4))                                         
      P0=DLAGRA_FHH_HSW(Z,AB,4,1)                                       
     $       
      PN=DLAGRA_FHH_HSW(Z(MAN),YZ,4,4)                                  
     $       
      GOTO 550                                                         
  540 P0=0.                                                            
      PN=0.                                                            
  550 PXZ(NR(I,J,1))=P0                                                
      PXZ(NR(I,J,NZ))=PN                                               
C                                                                      
      DO 520 K=1,NZ2                                                   
  520 B(K)=THREE*RLZ(K)/HZ(K)*(PX(NR(I,J,K+1))-PX(NR(I,J,K)))          
     & +THREE*RMUZ(K)/HZ(K+1)*(PX(NR(I,J,K+2))-PX(NR(I,J,K+1)))        
      B(1)=B(1)-RLZ(1)*P0                                              
      B(NZ2)=B(NZ2)-RMUZ(NZ2)*PN                                       
      CALL JORDAN_FHH_HSW(RMUZ,RLZ,XI,NZ2,B)                            
     $       
      DO 500 K=1,NZ2                                                   
  500 PXZ(NR(I,J,K+1))=XI(K)                                           
C                                                                      
C     SPLINE-FIT DE P(Y,Z)                                             
      DO 600 I=1,NX                                                    
      DO 600 J=1,NY                                                    
      GOTO (630,640),INX                                               
  630 DO 610 K=1,4                                                     
      AB(K)=PY(NR(I,J,K))                                              
  610 YZ(K)=PY(NR(I,J,K+NZ-4))                                         
      P0=DLAGRA_FHH_HSW(Z,AB,4,1)                                       
     $       
      PN=DLAGRA_FHH_HSW(Z(MAN),YZ,4,4)                                  
     $       
      GOTO 650                                                         
  640 P0=0.                                                            
      PN=0.                                                            
  650 PYZ(NR(I,J,1))=P0                                                
      PYZ(NR(I,J,NZ))=PN                                               
C                                                                      
      DO 620 K=1,NZ2                                                   
  620 B(K)=THREE*RLZ(K)/HZ(K)*(PY(NR(I,J,K+1))-PY(NR(I,J,K)))          
     & +THREE*RMUZ(K)/HZ(K+1)*(PY(NR(I,J,K+2))-PY(NR(I,J,K+1)))        
      B(1)=B(1)-RLZ(1)*P0                                              
      B(NZ2)=B(NZ2)-RMUZ(NZ2)*PN                                       
      CALL JORDAN_FHH_HSW(RMUZ,RLZ,XI,NZ2,B)                            
     $       
      DO 600 K=1,NZ2                                                   
  600 PYZ(NR(I,J,K+1))=XI(K)                                           
C                                                                      
C     SPLINE-FIT DE P(X,Y,Z)                                           
      DO 700 I=1,NX                                                    
      DO 700 J=1,NY                                                    
      GOTO (730,740),INX                                               
  730 DO 710 K=1,4                                                     
      AB(K)=PXY(NR(I,J,K))                                             
  710 YZ(K)=PXY(NR(I,J,K+NZ-4))                                        
      P0=DLAGRA_FHH_HSW(Z,AB,4,1)                                       
     $       
      PN=DLAGRA_FHH_HSW(Z(MAN),YZ,4,4)                                  
     $       
      GOTO 750                                                         
  740 P0=0.                                                            
      PN=0.                                                            
  750 PXYZ(NR(I,J,1))=P0                                               
      PXYZ(NR(I,J,NZ))=PN                                              
C                                                                      
      DO 720 K=1,NZ2                                                   
  720 B(K)=THREE*RLZ(K)/HZ(K)*(PXY(NR(I,J,K+1))-PXY(NR(I,J,K)))        
     & +THREE*RMUZ(K)/HZ(K+1)*(PXY(NR(I,J,K+2))-PXY(NR(I,J,K+1)))      
      B(1)=B(1)-RLZ(1)*P0                                              
      B(NZ2)=B(NZ2)-RMUZ(NZ2)*PN                                       
      CALL JORDAN_FHH_HSW(RMUZ,RLZ,XI,NZ2,B)                            
     $       
      DO 700 K=1,NZ2                                                   
  700 PXYZ(NR(I,J,K+1))=XI(K)                                          
      endif
      NXYZ=NX*NY*NZ                                                    
C                                                                      
C     SAVE GRID DEFINITION AND FUNCTIONS                               
C     WRITE (20) NX,(X(I),I=1,NX)                                      
C     WRITE (20) NY,(Y(I),I=1,NY)                                      
C     WRITE (20) NZ,(Z(I),I=1,NZ)                                      
C     WRITE (20) (F(I),I=1,NXYZ)                                       
C     WRITE (20) (PX(I),I=1,NXYZ)                                      
C     WRITE (20) (PY(I),I=1,NXYZ)                                      
C     WRITE (20) (PZ(I),I=1,NXYZ)                                      
C     WRITE (20) (PXY(I),I=1,NXYZ)                                     
C     WRITE (20) (PXZ(I),I=1,NXYZ)                                     
C     WRITE (20) (PYZ(I),I=1,NXYZ)                                     
C     WRITE (20) (PXYZ(I),I=1,NXYZ)                                    
C     close(20)
      RETURN                                                           
      END                                                              
C                                                                      
c                                                                      
c********************************************************************* 
      DOUBLE PRECISION FUNCTION DLAGRA_FHH_HSW(X,Y,MIN,IP)              
     $       
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)                               
      DIMENSION X(MIN),Y(MIN)                                          
      DLAGRA_FHH_HSW=0.D0                                               
     $       
      DO 10 I=1,MIN                                                    
      IF(I.EQ.IP) GOTO 10                                              
      YP=Y(I)                                                          
      DO 20 J=1,MIN                                                    
      IF(J.EQ.IP) GOTO 20                                              
      IF(J.EQ.I) GOTO 20                                               
      YP=YP*(X(IP)-X(J))                                               
   20 CONTINUE                                                         
      DO 30 J=1,MIN                                                    
      IF(J.EQ.I) GOTO 30                                               
      YP=YP/(X(I)-X(J))                                                
   30 CONTINUE                                                         
      DLAGRA_FHH_HSW=DLAGRA_FHH_HSW+YP                                  
     $               
   10 CONTINUE                                                         
      DO 40 I=1,MIN                                                    
      IF(I.EQ.IP) GOTO 40                                              
      DLAGRA_FHH_HSW=DLAGRA_FHH_HSW+Y(IP)/(X(IP)-X(I))                  
     $               
   40 CONTINUE                                                         
      RETURN                                                           
      END                                                              
c*************************************
      SUBROUTINE JORDAN_FHH_HSW(MU,LAMBDA,X,N,B)                        
     $       
      IMPLICIT DOUBLE PRECISION(A-H,L-M,O-Z)                           
c.....nmax should be max(nxmax,nymax,nzmax)
      PARAMETER (NMAX=100)
      DIMENSION MU(N),LAMBDA(N),X(N),PIV(NMAX),B(N)                     
C                                                                      
C     CALCUL DES PIVOTS                                                
      PIV(1)=2.D0                                                      
      DO 10 I=2,N                                                      
      PIV(I)=2.D0-LAMBDA(I)*MU(I-1)/PIV(I-1)                           
   10 B(I)=B(I)-LAMBDA(I)/PIV(I-1)*B(I-1)                              
C                                                                      
      X(N)=B(N)/PIV(N)                                                 
      I=N-1                                                            
   20 X(I)=(B(I)-X(I+1)*MU(I))/PIV(I)                                  
      I=I-1                                                            
      IF(I.GT.0) GOTO 20                                               
      RETURN                                                           
      END                                                              
c                                                                      

c*******************************************************************   
c      SUBROUTINE NATURL(R,INDX,FIN,*)                                  
      SUBROUTINE NATURL_FHH_HSW(R,INDX,FIN)                             
     $     
C                                                                      
C     INDX = 0  :  INTERPOLATED POTENTIAL ONLY                         
C            3  :        "      X,Y,Z-DERIVATIVES                      
C                                                                      
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)                               
      DIMENSION FIN(4),FIJK(64),HI(3),XR(3),U(3,4),R(3)                
      PARAMETER (NXMAX=40,NYMAX=40,NZMAX=40,NSMAX=NXMAX*NYMAX*NZMAX)
      COMMON /GRIDX_FHH_HSW/  X(NXMAX),NX                               
     $        
      COMMON /GRIDY_FHH_HSW/  Y(NYMAX),NY                               
     $        
      COMMON /GRIDZ_FHH_HSW/  Z(NZMAX),NZ                               
     $        
      DATA UN/1.D0/,TWO/2.D0/,THREE/3.D0/,SIX/6.D0/,IV,JV,KV/3*0/      
c check if point is inside grid;
c commented out since we do this test outside of this routine, 
c to make the calling routine more transparent. Therefore:
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c >>>>>>>> ATTENTION: this routine assumes that point R is inside the grid!!! c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cC                                                                      
cc      LS=1                                                             
cc      print*,R(1),X(1),X(NX)
cc      print*,R(2),Y(2),Y(NX)
cc      print*,R(3),Z(3),Z(NX)
c      IF(R(1).LT.X(1).OR.R(1).GT.X(NX)) GOTO 110                       
c      LS=2                                                             
c      IF(R(2).LT.Y(1).OR.R(2).GT.Y(NY)) GOTO 110                       
c      LS=3                                                             
c      IF(NZ.GT.1.and.(R(3).LT.Z(1).OR.R(3).GT.Z(NZ))) GOTO 110                       
      DO 10 IS=2,NX                                                    
      IF(R(1).LT.X(IS)) GOTO 20                                        
   10 CONTINUE                                                         
      is=nx
   20 DO 30 JS=2,NY                                                    
      IF(R(2).LT.Y(JS)) GOTO 40                                        
   30 CONTINUE                                                         
      js=ny
   40 DO 50 KS=2,NZ                                                    
      IF(R(3).LT.Z(KS)) GOTO 60                                        
   50 CONTINUE                                                         
      ks=max(2,nz)
   60 IF(IS.NE.IV) GOTO 65                                             
      IF(JS.NE.JV) GOTO 65                                             
      IF(KS.EQ.KV) GOTO 65                                             
   65 CALL FORMV_FHH_HSW(IS,JS,KS,FIJK)                                 
     $       
      HI(1)=X(IS)-X(IS-1)                                              
      IV=IS                                                            
      HI(2)=Y(JS)-Y(JS-1)                                              
      JV=JS                                                            
      if(nz.gt.1)HI(3)=Z(KS)-Z(KS-1)                                    
      if(nz.eq.1)HI(3)=1.d20
      KV=KS                                                            
   75 XR(1)=(R(1)-X(IS-1))/HI(1)                                       
      XR(2)=(R(2)-Y(JS-1))/HI(2)                                       
      XR(3)=(R(3)-Z(KS-1))/HI(3)                                       
      IN=INDX                                                          
   90 DO 70 I=1,3                                                      
      IF(I.EQ.IN) GOTO 70                                              
      XR2=XR(I)**2                                                     
      XR3=XR(I)*XR2                                                    
      U(I,1)=-TWO*XR3+THREE*XR2                                        
      U(I,2)=-U(I,1)+UN                                                
      U(I,3)=HI(I)*(XR3-XR2)                                           
      U(I,4)=HI(I)*(XR3-TWO*XR2+XR(I))                                 
   70 CONTINUE                                                         
      IF(IN.EQ.0) GOTO 80                                              
      U(IN,1)=SIX*(-XR(IN)+1)*XR(IN)/HI(IN)                            
      U(IN,2)=-U(IN,1)                                                 
      U(IN,3)=(THREE*XR(IN)-TWO)*XR(IN)                                
      U(IN,4)=U(IN,3)-TWO*XR(IN)+UN                                    
   80 FIN(IN+1)=0.D0                                                   
      IJK=1                                                            
      DO 100 K=1,4                                                     
      DO 100 J=1,4                                                     
      UKJ=U(3,K)*U(2,J)                                                
      DO 100 I=1,4                                                     
      FIN(IN+1)=FIN(IN+1)+FIJK(IJK)*UKJ*U(1,I)                         
  100 IJK=IJK+1                                                        
      IN=IN-1                                                          
      IF(IN.GT.0.AND.INDX.GT.0) GOTO 90                                
      RETURN                                                           
c  110 continue
c point at R() is off-grid; but re
c      PRINT 120,LS                                                     
c       print*,R(1),R(2),R(3)
c  120 FORMAT( 1X,15('-'),' R(',I1,') SORT DE LA GRILLE')               
c      RETURN 1                                                         
      END                                                              
c                                                                      
c**********************************************************************
      SUBROUTINE FORMV_FHH_HSW(I,J,K,G)                                 
     $       
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)                               
      DIMENSION G(64)                                                  
      PARAMETER (NXMAX=40,NYMAX=40,NZMAX=40,NSMAX=NXMAX*NYMAX*NZMAX)
      COMMON /SURFAC_FHH_HSW/ F(NSMAX),                                 
     $        
     & PX(NSMAX),PY(NSMAX),PZ(NSMAX),                                   
     & PXY(NSMAX),PXZ(NSMAX),PYZ(NSMAX),                                
     & PXYZ(NSMAX)                                                      
      COMMON /GRIDX_FHH_HSW/  X(NXMAX),NX                               
     $        
      COMMON /GRIDY_FHH_HSW/  Y(NYMAX),NY                               
     $        
      COMMON /GRIDZ_FHH_HSW/  Z(NZMAX),NZ                               
     $        
C                                                                      
      IJK=(K-1)*NX*NY+(J-1)*NX+I                                       
      IMJK=IJK-1                                                       
      IJMK=IJK-NX                                                      
      IMJMK=IJMK-1                                                     
      IJKM=IJK-NX*NY                                                   
      IMJKM=IJKM-1                                                     
      IJMKM=IJKM-NX                                                    
      IMJMKM=IJMKM-1                                                   
C                                                                      
      G( 1)=F(IJK)                                                     
      G( 2)=F(IMJK)                                                    
      G( 3)=PX(IJK)                                                    
      G( 4)=PX(IMJK)                                                   
      G( 5)=F(IJMK)                                                    
      G( 6)=F(IMJMK)                                                   
      G( 7)=PX(IJMK)                                                   
      G( 8)=PX(IMJMK)                                                  
      G( 9)=PY(IJK)                                                    
      G(10)=PY(IMJK)                                                   
      G(11)=PXY(IJK)                                                   
      G(12)=PXY(IMJK)                                                  
      G(13)=PY(IJMK)                                                   
      G(14)=PY(IMJMK)                                                  
      G(15)=PXY(IJMK)                                                  
      G(16)=PXY(IMJMK)                                                 
      G(17)=F(IJKM)                                                    
      G(18)=F(IMJKM)                                                   
      G(19)=PX(IJKM)                                                   
      G(20)=PX(IMJKM)                                                  
      G(21)=F(IJMKM)                                                   
      G(22)=F(IMJMKM)                                                  
      G(23)=PX(IJMKM)                                                  
      G(24)=PX(IMJMKM)                                                 
      G(25)=PY(IJKM)                                                   
      G(26)=PY(IMJKM)                                                  
      G(27)=PXY(IJKM)                                                  
      G(28)=PXY(IMJKM)                                                 
      G(29)=PY(IJMKM)                                                  
      G(30)=PY(IMJMKM)                                                 
      G(31)=PXY(IJMKM)                                                 
      G(32)=PXY(IMJMKM)                                                
      G(33)=PZ(IJK)                                                    
      G(34)=PZ(IMJK)                                                   
      G(35)=PXZ(IJK)                                                   
      G(36)=PXZ(IMJK)                                                  
      G(37)=PZ(IJMK)                                                   
      G(38)=PZ(IMJMK)                                                  
      G(39)=PXZ(IJMK)                                                  
      G(40)=PXZ(IMJMK)                                                 
      G(41)=PYZ(IJK)                                                   
      G(42)=PYZ(IMJK)                                                  
      G(43)=PXYZ(IJK)                                                  
      G(44)=PXYZ(IMJK)                                                 
      G(45)=PYZ(IJMK)                                                  
      G(46)=PYZ(IMJMK)                                                 
      G(47)=PXYZ(IJMK)                                                 
      G(48)=PXYZ(IMJMK)                                                
      G(49)=PZ(IJKM)                                                   
      G(50)=PZ(IMJKM)                                                  
      G(51)=PXZ(IJKM)                                                  
      G(52)=PXZ(IMJKM)                                                 
      G(53)=PZ(IJMKM)                                                  
      G(54)=PZ(IMJMKM)                                                 
      G(55)=PXZ(IJMKM)                                                 
      G(56)=PXZ(IMJMKM)                                                
      G(57)=PYZ(IJKM)                                                  
      G(58)=PYZ(IMJKM)                                                 
      G(59)=PXYZ(IJKM)                                                 
      G(60)=PXYZ(IMJKM)                                                
      G(61)=PYZ(IJMKM)                                                 
      G(62)=PYZ(IMJMKM)                                                
      G(63)=PXYZ(IJMKM)                                                
      G(64)=PXYZ(IMJMKM)                                               
      RETURN                                                           
      end                                                              
c ENDE DES 3DPLINE-PROGRAMMS!                                                                                                       




	subroutine potread(ipes)
c   ipes=1 for CBS
c   ipes=2 for av5z
c   in bondlength coordinates (rich1,rch2,rhh)
      implicit real*8(a-h,o-z)
	
      parameter (pi=3.141592653589793d0)    
      parameter (nhh=80,nfh=30,nth=19)
	parameter (m=100)
	dimension xt(m)  
       dimension y2(m),y(m)
        data vmin1/-100.83635116d0/
       data dy1,dyn/1.0d30,1.0d30/
      common /pes/rhh(nth,nfh,nhh),rfh(nfh),thth(nth),vcut,
     & ve(nth,nfh,nhh),ind(nth,nfh)
      open(98,file='fh2av5z-final.txt')
c      open(98,file='zz.txt')
	open(66,file='check.txt')
	vcut=6.0d0/27.2116080d0
        read(98,*)
        read(98,*)
	 ntt=0
      do i=1,nth
	 do j=1,nfh
	  do k=1,nhh
	 ntt=ntt+1
          read(98,*)th1,ra1,ra2,ve1,ve2
          if (abs(th1+ra1+ra2).lt.0.0001)goto 11
	  if (k.eq.1) then
	    th1aa=th1
	    ra1aa=ra1
	  endif
       if (abs(th1-th1aa).gt.0.001.or.abs(ra1aa-ra1).gt.0.001) then
	 write(*,*)' error',' ntt=',ntt,th1,ra1
	 endif
          thth(i)=th1
	  rhh(i,j,k)=ra2
	  rfh(j)=ra1
	  if (k.gt.1.and.ra2.lt.rhh(i,j,k-1)) then
	   write(*,*)' error',' ntt=',ntt,ra2
	 endif
	  if (ipes.eq.1) then
	  ve(i,j,k)=ve1-vmin1-0.000211642d0
	  endif

	  if (ipes.eq.2) then
	  ve(i,j,k)=ve2-vmin1-.00021380093d0
	  endif

        enddo
11         continue

           kk=k-1
           kkk=kk
           ind(i,j)=kk
        enddo;enddo
c	write(*,*)' th=',(thth(i),i=1,nth)
c	write(*,*)' rfh=',(rfh(i),i=1,nfh)
c        write(*,*)' nth=',nth,' nfh=',nfh
       close(98)
c	  write(*,*)' check'
	 nadd=0
	 na=0
 	do i=1,nth
	do j=1,nfh
	na=na+1
	do k=1,ind(i,j)
      	write(66,1001)thth(i),rfh(j),rhh(i,j,k),
     &      ve(i,j,k)*27.2116080d0
 1001   format (1x,1f15.8,2f6.2,1f15.8)
	enddo
	enddo
	enddo

      end









       subroutine spl3(r1,r2,th,v,iop)
       implicit real*8(a-h,o-z)
        parameter (nhh=80,nfh=30,nth=19)
	parameter (m=100,n=100,mm=100)
	dimension xt(m)  
       dimension dty(mm),ddty(mm),s1(mm),ds1(mm),dds1(mm),h1(mm)
       dimension dny(n),ddny(n),s2(n),ds2(n),dds2(n),h2(n)
       dimension dhy(m),ddhy(m),s3(m),ds3(m),dds3(m),h3(m)
       dimension y(m),ss(m),sss(m),y2(m)
	dimension xt0(m),y0(m)
      common /pes/rhh(nth,nfh,nhh),rfh(nfh),thth(nth),vcut,
     & ve(nth,nfh,nhh),ind(nth,nfh)
       data dy1,dyn/1.0d30,1.0d30/
c	write(*,*)' r1=',r1,r2,th
       do 20 i=1,nth
       do 10 j=1,nfh
	dmin=1000
  	r3a=r1
	if (r3a.lt.rhh(i,j,1)) then
          r3a=rhh(i,j,1)
	  y3=vcut+0.0005
	  goto 28
	endif
	if (r3a.gt.rhh(i,j,ind(i,j))) r3a=rhh(i,j,ind(i,j))
	do k=1,ind(i,j)
	dd=abs(r3a-rhh(i,j,k))
	if (dd.lt.dmin) then
	dmin=dd
	km=k
	endif
	enddo

	kmin=1
	kmax=ind(i,j)
	do k=km,1,-1
	if (ve(i,j,k).gt.vcut) then
	if ((ve(i,j,k)-ve(i,j,k+1))*27.2116.gt.0.5) then
	kmin=k+1
	else
	kmin=k
	endif
	goto 41
	endif
	enddo
 41	continue
	do k=km+1,ind(i,j)
	if (ve(i,j,k).gt.vcut) then
	if ((ve(i,j,k)-ve(i,j,k-1))*27.2116.gt.0.5) then
	kmax=k-1
	else 
        kmax=k
	endif
	goto 42
	endif
	enddo
 42	continue
	
	nh=0
       do 2 k=kmin,kmax
	nh=nh+1
	xt(nh)=rhh(i,j,k)
	y(nh)=ve(i,j,k)
   2   continue
	if (nh.le.3) then
	y3=vcut+0.0005
	goto 28
	endif
	
        if (r3a.gt.xt(nh)) r3a=xt(nh)
        if (r3a.lt.xt(1)) then
        y3=vcut+0.0005
        r3a=xt(1)
	goto 28
	endif
c	write(*,*)i,j,kmin,kmax,nh
c	write(*,*)' r3a=',r3a
c	write(*,*)(y(ii)*27.212,ii=1,nh)
	call xie_spline(xt,y,nh,dy1,dyn,y2)
        call xie_splint(xt,y,y2,nh,r3a,y3)
 28     continue
        ss(j)=y3
   10   continue
         
c	write(*,*)' ss',(ss(ii)*27.212,ii=1,nfh)
	nr=0
       do 5 j=1,nfh
	if (ss(j).gt.vcut.and.nr.eq.0) then
	else
	nr=nr+1
	xt(nr)=rfh(j)
        y(nr)=ss(j)
	if (y(nr).gt.vcut) y(nr)=vcut
	endif
   5   continue
c	write(*,*)' nr=',nr,' r2=',r2
c	write(*,*)' ss',(ss(ii)*27.212,ii=1,nr)
c	write(*,*)' y',(y(ii)*27.212,ii=1,nr)
c	write(*,*)' xt',(xt(ii),ii=1,nr)
	r2a=r2
	if (r2a.lt.xt(1).or.nr.le.2) then
	yw2=vcut
	goto 19
	endif
	if (r2a.gt.xt(nr)) r2a=xt(nr)
	call xie_spline(xt,y,nr,dy1,dyn,y2)
	call xie_splint(xt,y,y2,nr,r2a,yw2)
 19	continue
       sss(i)=yw2
c	write(*,*)' yw2=',yw2
   20    continue
c	write(*,*)' sss=',(sss(i)*27.212,i=1,nth)
	nr=0
	do i=1,nth
	nr=nr+1
	xt(nr)=thth(i)
	y(nr)=sss(i)
  	enddo
c	write(*,*)' sss',(sss(ii)*27.212,ii=1,nr)
	call xie_spline(xt,y,nr,0.0d0,0.0d0,y2)
	call xie_splint(xt,y,y2,nr,th,yw2)
   44	continue
        v=yw2
	if (v.gt.vcut) v=vcut
       end

	   SUBROUTINE ESPL2(X,Y,N,DY1,DYN,XX,M,DY,DDY,S,DS,DDS,T,H)
           parameter (nmax=100)
!           DIMENSION X(N),Y(N),XX(M),DY(N),DDY(N)
!        DIMENSION S(M),DS(M),DDS(M),H(N)
           dimension x(nmax),y(nmax),xx(nmax),dy(nmax),ddy(nmax)
           dimension s(nmax),ds(nmax),dds(nmax),h(nmax)
	   DOUBLE PRECISION X,Y,XX,DY,DDY,S,DS,DDS,H,DY1,DYN,
     *                   T,H0,H1,BETA,ALPHA
c	   DY(1)=-0.5
	   H0=X(2)-X(1)
c	   H(1)=3.0*(Y(2)-Y(1))/(2.0*H0)-DY1*H0/4.0
		dy(1)=0
		h(1)=0 
	   DO 10 J=2,N-1
	  H1=X(J+1)-X(J)
	  ALPHA=H0/(H0+H1)
	  BETA=(1.0-ALPHA)*(Y(J)-Y(J-1))/H0
	  BETA=3.0*(BETA+ALPHA*(Y(J+1)-Y(J))/H1)
	  DY(J)=-ALPHA/(2.0+(1.0-ALPHA)*DY(J-1))
	  H(J)=(BETA-(1.0-ALPHA)*H(J-1))
	  H(J)=H(J)/(2.0+(1.0-ALPHA)*DY(J-1))
	  H0=H1
10	  CONTINUE
c	  DY(N)=(3.0*(Y(N)-Y(N-1))/H1+DYN*H1/2.0-H(N-1))
c     *         /(2.0+DY(N-1))
	  DY(N)=0
	  DO 20 J=N-1,1,-1
20	  DY(J)=DY(J)*DY(J+1)+H(J)
	  DO 30 J=1,N-1
30	  H(J)=X(J+1)-X(J)
	  DO 40 J=1,N-1
	  H1=H(J)*H(J)
	  DDY(J)=6.0*(Y(J+1)-Y(J))/H1-
     *           2.0*(2.0*DY(J)+DY(J+1))/H(J)
40	  CONTINUE
	  H1=H(N-1)*H(N-1)
	  DDY(N)=6.0*(Y(N-1)-Y(N))/H1+
     *            2.0*(2.0*DY(N)+DY(N-1))/H(N-1)
	  T=0.0
	  DO 50 I=1,N-1
	  H1=0.5*H(I)*(Y(I)+Y(I+1))
	  H1=H1-H(I)*H(I)*H(I)*(DDY(I)+DDY(I+1))/24.0
	  T=T+H1
50	  CONTINUE
	  DO 70 J=1,M
	  IF (XX(J).GE.X(N)) THEN
	    I=N-1
	  ELSE
	    I=1
60	    IF (XX(J).GT.X(I+1)) THEN
	      I=I+1
	      GOTO 60
	    END IF
	  END IF
	  H1=(X(I+1)-XX(J))/H(I)
	  S(J)=(3.0*H1*H1-2.0*H1*H1*H1)*Y(I)
	  S(J)=S(J)+H(I)*(H1*H1-H1*H1*H1)*DY(I)
	  DS(J)=6.0*(H1*H1-H1)*Y(I)/H(I)
	  DS(J)=DS(J)+(3.0*H1*H1-2.0*H1)*DY(I)
	  DDS(J)=(6.0-12.0*H1)*Y(I)/(H(I)*H(I))
	  DDS(J)=DDS(J)+(2.0-6.0*H1)*DY(I)/H(I)
	  H1=(XX(J)-X(I))/H(I)
	  S(J)=S(J)+(3.0*H1*H1-2.0*H1*H1*H1)*Y(I+1)
	  S(J)=S(J)-H(I)*(H1*H1-H1*H1*H1)*DY(I+1)
	  DS(J)=DS(J)-6.0*(H1*H1-H1)*Y(I+1)/H(I)
	  DS(J)=DS(J)+(3.0*H1*H1-2.0*H1)*DY(I+1)
	  DDS(J)=DDS(J)+(6.0-12.0*H1)*Y(I+1)/(H(I)*H(I))
	  DDS(J)=DDS(J)-(2.0-6.0*H1)*DY(I+1)/H(I)
70	  CONTINUE
	  RETURN
	  END


C##################################################################
C# SPLINE ROUTINES
C#            Numerical recipes in fortran
C#            Cambrige University Press
C#            York, 2nd edition, 1992.
C##################################################################
      SUBROUTINE xie_splint(xa,ya,y2a,n,x,y)
      implicit double precision  (a-h,o-z)
      DIMENSION xa(n),y2a(n),ya(n)
      klo=1
      khi=n
 1    if (khi-klo.gt.1) then
        k=(khi+klo)/2
        if(xa(k).gt.x)then
          khi=k
        else
          klo=k
        endif
      goto 1
      endif
      h=xa(khi)-xa(klo)
      if (h.eq.0.0d0) then
	write(*,*)' error h=',0
	write(*,*)' number of point=',n
	return
	endif
      a=(xa(khi)-x)/h
      b=(x-xa(klo))/h
      y=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**
     &2)/6.0d0
      return
      END
C##############################################################################
      SUBROUTINE xie_spline(x,y,n,yp1,ypn,y2)
      implicit double precision  (a-h,o-z)
      DIMENSION x(n),y(n),y2(n)
      PARAMETER (NMAX=100)
      DIMENSION u(NMAX)
      if (yp1.gt..99d30) then
        y2(1)=0.0d0
        u(1)=0.0d0
      else
        y2(1)=-0.5d0
        u(1)=(3.0d0/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
      endif
      do 11 i=2,n-1
        sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
        p=sig*y2(i-1)+2.0d0
        y2(i)=(sig-1.0d0)/p
        u(i)=(6.0d0*((y(i+1)-y(i))/(x(i+
     *1)-x(i))-(y(i)-y(i-1))/(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*
     *u(i-1))/p
11    continue
      if (ypn.gt..99d30) then
        qn=0.0d0
        un=0.0d0
      else
        qn=0.5d0
        un=(3.0d0/(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
      endif
      y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.0d0)
      do 12 k=n-1,1,-1
        y2(k)=y2(k)*y2(k+1)+u(k)
12    continue
      return
      END



	subroutine jacobitobond(r1a,r2a,ctha,rrhh,rrhf,cthb)
        implicit real*8(a-h,o-z)
        ra=r2a/2.0d0
        rrhh=r2a
        rrhf=dsqrt(r1a*r1a+ra*ra-2.0d0*r1a*ra*ctha)
        cthb=(rrhf*rrhf+ra*ra-r1a*r1a)/(2.0d0*rrhf*ra)
        if(cthb.gt.1)cthb=1
        if(cthb.lt.-1)cthb=-1
        end



	subroutine jacobitobond2(r1a,r2a,ctha,rrhh,rrhf,cthb)
        implicit real*8(a-h,o-z)
        ra=r2a/2.0d0
        rrhf=r2a
        rrhh=dsqrt(r1a*r1a+ra*ra-2.0d0*r1a*ra*ctha)
        cthb=(rrhh*rrhh+ra*ra-r1a*r1a)/(2.0d0*rrhh*ra)
        if(cthb.gt.1)cthb=1
        if(cthb.lt.-1)cthb=-1
        end


	subroutine extrap(v3,v4,v5,vcbs)
	implicit real*8(a-h,o-z)
c    V(n)=vcbs+a*exp(-n)+b*exp(-n*n)
c     v3 for avtz, v4 for a4qz, v5 for av5z
	c1=dexp(-3.0d0)-dexp(-4.0d0)
	c2=dexp(-9.0d0)-dexp(-16.0d0)
	c3=dexp(-4.0d0)-dexp(-5.0d0)
	c4=dexp(-16.0d0)-dexp(-25.0d0)
	a=(c4*(v3-v4)-c2*(v4-v5))/(c4*c1-c2*c3)
	b=(v3-v4-a*c1)/c2
	vcbs=v3-a*dexp(-3.d0)-b*dexp(-9.0d0)
	vcbs1=v4-a*dexp(-4.d0)-b*dexp(-16.0d0)
	vcbs2=v5-a*dexp(-5.d0)-b*dexp(-25.0d0)
	write(*,*)vcbs,vcbs1,vcbs2
	end

	subroutine extrap1(v3,v4,v5,vcbs)
	implicit real*8(a-h,o-z)
c  V(n)=Vcbs+B*Exp(-cn)
c     v3 for avtz, v4 for a4qz, v5 for av5z
	alpha=(v3-v4)/(v4-v5)
	x=1/alpha
	b=(v3-v4)/(x**3-x**4)
	vcbs=v3-b*x**3
	vcbs1=v4-b*x**4
	vcbs2=v5-b*x**5
	write(*,*)vcbs,vcbs1,vcbs2
	end
	
	subroutine extrap2(v3,v4,v5,vcbs)
	implicit real*8(a-h,o-z)
c    V(n)=vcbs+a*exp(-n(n-1)+b*exp(-(n-1)*(n-1))
c     v3 for avtz, v4 for a4qz, v5 for av5z
	 vcbs=(5**3*v5-4**3*v4)/(5**3-4**3)
	end
	subroutine extrap3(v3,v4,v5,vcbs)
	implicit real*8(a-h,o-z)
c    V(n)=vcbs+a*exp(-(n-1))+b*exp((n-1)*(n-1))
c     v3 for avtz, v4 for a4qz, v5 for av5z
	c1=dexp(-2.0d0)-dexp(-3.0d0)
	c2=dexp(-4.0d0)-dexp(-9.0d0)
	c3=dexp(-3.0d0)-dexp(-4.0d0)
	c4=dexp(-9.0d0)-dexp(-16.0d0)
	a=(c4*(v3-v4)-c2*(v4-v5))/(c4*c1-c2*c3)
	b=(v3-v4-a*c1)/c2
	vcbs=v3-a*dexp(-2.d0)-b*dexp(-4.d0)
	vcbs1=v4-a*dexp(-3.d0)-b*dexp(-9.d0)
	vcbs2=v5-a*dexp(-4.d0)-b*dexp(-16.d0)
	write(*,*)vcbs,vcbs1,vcbs2
	end

c=========================================================
        function ItoC(ID0)
        character*6 ITOC
        character*1 CH

	ID=ID0
	N1=10; N2=1
	DO I=6,1,-1
	  N=MOD(ID,N1); N=(N/N2)
          ITOC(I:I)=CHAR(48+N)
	  N1=N1*10; N2=N2*10
	END DO

        RETURN
        END

