      subroutine prepar(nx,ny,nz,x,y,z,en)
      PARAMETER (NXMAX=40,NYMAX=40,NZMAX=40)
      real*8 x(NXMAX),y(NYMAX),z(NZMAX),en(NXMAX,NYMAX,NZMAX)
      Include 'hswdata.inc'
      open(unit=10,file=DataDir(1:DataDirLength)//'/so.input')
      read(10,*) nx,ny,nz
      do 10 i=1,nx
         do 11 j=1,ny
            do 12 k=1,nz
               read(10,*) x(i),y(j),z(k),en(i,j,k)
   12       continue
   11    continue
   10 continue
c
c.....generate spline fit
c
      call spmain(nx,ny,nz,x,y,z,en)
      return
      end
c------------------------------------------------------
      subroutine spmain(nax,nay,naz,xa,ya,za,ea)                              
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)                               
      PARAMETER (NXMAX=40,NYMAX=40,NZMAX=40,NSMAX=NXMAX*NYMAX*NZMAX)
      real*8 xa(NXMAX),ya(NYMAX),za(NZMAX),ea(NXMAX,NYMAX,NZMAX)
      COMMON /SURFAC/ F(NSMAX),                                                
     & PX(NSMAX),PY(NSMAX),PZ(NSMAX),                                     
     & PXY(NSMAX),PXZ(NSMAX),PYZ(NSMAX),                                  
     & PXYZ(NSMAX)                                                      
      COMMON /GRIDX/  X(NXMAX),NX                                             
      COMMON /GRIDY/  Y(NYMAX),NY                                             
      COMMON /GRIDZ/  Z(NZMAX),NZ                                             
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
      CALL SPLINF(2)                                                   
	return
      END                                                              
c                                                                      
c********************************************************************* 
      SUBROUTINE SPLINF(INX)                                           
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)                               
      PARAMETER (NXMAX=40,NYMAX=40,NZMAX=40,NSMAX=NXMAX*NYMAX*NZMAX)
      common/SPF/ HX(NXMAX),HY(NXMAX),HZ(NXMAX),RLX(NXMAX),RMUX(NXMAX),
     & RLY(NXMAX),RMUY(NXMAX),RLZ(NXMAX),RMUZ(NXMAX),XI(NXMAX),B(NXMAX),
     & AB(4),YZ(4)     
      COMMON /SURFAC/ F(NSMAX),                                                
     & PX(NSMAX),PY(NSMAX),PZ(NSMAX),                                     
     & PXY(NSMAX),PXZ(NSMAX),PYZ(NSMAX),                                  
     & PXYZ(NSMAX)                                                      
      COMMON /GRIDX/  X(NXMAX),NX                                             
      COMMON /GRIDY/  Y(NYMAX),NY                                             
      COMMON /GRIDZ/  Z(NZMAX),NZ                                             
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
      P0=DLAGRA(X,AB,4,1)                                              
      PN=DLAGRA(X(MAN),YZ,4,4)                                         
      PX(NR(1,J,K))=P0                                                 
      PX(NR(NX,J,K))=PN                                                
C     CALCUL SECOND MEMBRE                                             
      DO 120 I=1,NX2                                                   
  120 B(I)=THREE*RLX(I)/HX(I)*(F(NR(I+1,J,K))-F(NR(I,J,K)))            
     & +THREE*RMUX(I)/HX(I+1)*(F(NR(I+2,J,K))-F(NR(I+1,J,K)))          
      B(1)=B(1)-RLX(1)*P0                                              
      B(NX2)=B(NX2)-RMUX(NX2)*PN                                       
      CALL JORDAN(RMUX,RLX,XI,NX2,B)                                   
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
      P0=DLAGRA(Y,AB,4,1)                                              
      PN=DLAGRA(Y(MAN),YZ,4,4)                                         
      PY(NR(I,1,K))=P0                                                 
      PY(NR(I,NY,K))=PN                                                
C                                                                      
      DO 220 J=1,NY2                                                   
  220 B(J)=THREE*RLY(J)/HY(J)*(F(NR(I,J+1,K))-F(NR(I,J,K)))            
     & +THREE*RMUY(J)/HY(J+1)*(F(NR(I,J+2,K))-F(NR(I,J+1,K)))          
      B(1)=B(1)-RLY(1)*P0                                              
      B(NY2)=B(NY2)-RMUY(NY2)*PN                                       
      CALL JORDAN(RMUY,RLY,XI,NY2,B)                                   
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
      P0=DLAGRA(Z,AB,4,1)                                              
      PN=DLAGRA(Z(MAN),YZ,4,4)                                         
      PZ(NR(I,J,1))=P0                                                 
      PZ(NR(I,J,NZ))=PN                                                
C                                                                      
      DO 320 K=1,NZ2                                                   
  320 B(K)=THREE*RLZ(K)/HZ(K)*(F(NR(I,J,K+1))-F(NR(I,J,K)))            
     & +THREE*RMUZ(K)/HZ(K+1)*(F(NR(I,J,K+2))-F(NR(I,J,K+1)))          
      B(1)=B(1)-RLZ(1)*P0                                              
      B(NZ2)=B(NZ2)-RMUZ(NZ2)*PN                                       
      CALL JORDAN(RMUZ,RLZ,XI,NZ2,B)                                   
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
      P0=DLAGRA(Y,AB,4,1)                                              
      PN=DLAGRA(Y(MAN),YZ,4,4)                                         
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
      CALL JORDAN(RMUY,RLY,XI,NY2,B)                                   
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
      P0=DLAGRA(Z,AB,4,1)                                              
      PN=DLAGRA(Z(MAN),YZ,4,4)                                         
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
      CALL JORDAN(RMUZ,RLZ,XI,NZ2,B)                                   
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
      P0=DLAGRA(Z,AB,4,1)                                              
      PN=DLAGRA(Z(MAN),YZ,4,4)                                         
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
      CALL JORDAN(RMUZ,RLZ,XI,NZ2,B)                                   
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
      P0=DLAGRA(Z,AB,4,1)                                              
      PN=DLAGRA(Z(MAN),YZ,4,4)                                         
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
      CALL JORDAN(RMUZ,RLZ,XI,NZ2,B)                                   
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
      DOUBLE PRECISION FUNCTION DLAGRA(X,Y,MIN,IP)                     
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)                               
      DIMENSION X(MIN),Y(MIN)                                          
      DLAGRA=0.D0                                                      
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
      DLAGRA=DLAGRA+YP                                                 
   10 CONTINUE                                                         
      DO 40 I=1,MIN                                                    
      IF(I.EQ.IP) GOTO 40                                              
      DLAGRA=DLAGRA+Y(IP)/(X(IP)-X(I))                                 
   40 CONTINUE                                                         
      RETURN                                                           
      END                                                              
c*************************************
      SUBROUTINE JORDAN(MU,LAMBDA,X,N,B)                               
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
      SUBROUTINE NATURL(R,INDX,FIN)                                  
C                                                                      
C     INDX = 0  :  INTERPOLATED POTENTIAL ONLY                         
C            3  :        "      X,Y,Z-DERIVATIVES                      
C                                                                      
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)                               
      DIMENSION FIN(4),FIJK(64),HI(3),XR(3),U(3,4),R(3)                
      PARAMETER (NXMAX=40,NYMAX=40,NZMAX=40,NSMAX=NXMAX*NYMAX*NZMAX)
      COMMON /GRIDX/  X(NXMAX),NX                                             
      COMMON /GRIDY/  Y(NYMAX),NY                                             
      COMMON /GRIDZ/  Z(NZMAX),NZ                                             
      DATA UN/1.D0/,TWO/2.D0/,THREE/3.D0/,SIX/6.D0/,IV,JV,KV/3*0/      
c check if point is inside grid;
c commented out since we do this test outside of this routine, 
c to make the calling routine more transparent. Therefore:
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c >>>>>>>> ATTENTION: this routine assumes that point R is inside the grid!!! c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cC                                                                      
cc      LS=1                                                             
cc	print*,R(1),X(1),X(NX)
cc	print*,R(2),Y(2),Y(NX)
cc	print*,R(3),Z(3),Z(NX)
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
   65 CALL FORMV(IS,JS,KS,FIJK)                                        
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
c	print*,R(1),R(2),R(3)
c  120 FORMAT( 1X,15('-'),' R(',I1,') SORT DE LA GRILLE')               
c      RETURN 1                                                         
      END                                                              
c                                                                      
c**********************************************************************
      SUBROUTINE FORMV(I,J,K,G)                                        
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)                               
      DIMENSION G(64)                                                  
      PARAMETER (NXMAX=40,NYMAX=40,NZMAX=40,NSMAX=NXMAX*NYMAX*NZMAX)
      COMMON /SURFAC/ F(NSMAX),                                                
     & PX(NSMAX),PY(NSMAX),PZ(NSMAX),                                     
     & PXY(NSMAX),PXZ(NSMAX),PYZ(NSMAX),                                  
     & PXYZ(NSMAX)                                                      
      COMMON /GRIDX/  X(NXMAX),NX                                             
      COMMON /GRIDY/  Y(NYMAX),NY                                             
      COMMON /GRIDZ/  Z(NZMAX),NZ                                             
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


