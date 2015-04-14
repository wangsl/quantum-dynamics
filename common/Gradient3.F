
C     $Id$

C     Reference
C     http://www.trentfguidry.net/post/2010/09/04/Numerical-differentiation-formulas.aspx

#include "matutils.h"

      SubRoutine Gradient3D(N1, N2, N3, N2P, Dx, F, V, G, NPoints)
      Integer N1, N2, N3, N2P, NPoints
      Real*8 Dx
      Complex*16 F(N1,N2,N3), V(N1,N3), G(N1,N3)

      If(NPoints .eq. 3) then
         Call Gradient3(N1, N2, N3, N2P, Dx, F, V, G)
      Else If(NPoints .eq. 5) then
         Call Gradient5(N1, N2, N3, N2P, Dx, F, V, G)
      Else If(NPoints .eq. 7) then
         Call Gradient7(N1, N2, N3, N2P, Dx, F, V, G)
      Else If(NPoints .eq. 9) then
         Call Gradient9(N1, N2, N3, N2P, Dx, F, V, G)
      Else If(NPoints .eq. 11) then
         Call Gradient11(N1, N2, N3, N2P, Dx, F, V, G)
      Else
         Call MCrash('NPoints error')
      EndIf

      End


      SubRoutine Gradient3(N1, N2, N3, N2P, Dx, F, V, G)
      Implicit None
      Integer N1, N2, N3, N2P
      Real*8 Dx
      Complex*16 F(N1,N2,N3), V(N1,N3), G(N1,N3)
      
      Integer I, K
      
C$OMP Parallel Do
C$OMP& Schedule(Static, 1)
C$OMP& Default(Shared)
C$OMP& Private(I, K)
      Do K = 1, N3
         Do I = 1, N1
            V(I, K) = F(I, N2P, K)
            
            G(I,K) = 
     $           + (F(I,N2P+1,K) - F(I,N2P-1,K)) 
            G(I,K) = G(I,K)/(2*Dx)
         EndDo
      EndDo
C$OMP End Parallel Do
      Return
      End


      SubRoutine Gradient5(N1, N2, N3, N2P, Dx, F, V, G)
      Implicit None
      Integer N1, N2, N3, N2P
      Real*8 Dx
      Complex*16 F(N1,N2,N3), V(N1,N3), G(N1,N3)
      
      Integer I, K
      
C$OMP Parallel Do
C$OMP& Schedule(Static, 1)
C$OMP& Default(Shared)
C$OMP& Private(I, K)
      Do K = 1, N3
         Do I = 1, N1
            V(I, K) = F(I, N2P, K)
            
            G(I,K) = 
     $           - (F(I,N2P+2,K) - F(I,N2P-2,K)) 
     $           + 8*(F(I,N2P+1,K) - F(I,N2P-1,K)) 
            G(I,K) = G(I,K)/(12*Dx)
         EndDo
      EndDo
C$OMP End Parallel Do
      Return
      End

      SubRoutine Gradient7(N1, N2, N3, N2P, Dx, F, V, G)
      Implicit None
      Integer N1, N2, N3, N2P
      Real*8 Dx
      Complex*16 F(N1,N2,N3), V(N1,N3), G(N1,N3)
      
      Integer I, K

C$OMP Parallel Do
C$OMP& Schedule(Static, 1)
C$OMP& Default(Shared)
C$OMP& Private(I, K)
      Do K = 1, N3
         Do I = 1, N1
            V(I, K) = F(I, N2P, K)
            
            G(I,K) = 
     $           + (F(I,N2P+3,K) - F(I,N2P-3,K)) 
     $           - 9*(F(I,N2P+2,K) - F(I,N2P-2,K)) 
     $           + 45*(F(I,N2P+1,K) - F(I,N2P-1,K)) 
            G(I,K) = G(I,K)/(60*Dx)
         EndDo
      EndDo
C$OMP End Parallel Do
      Return
      End


      SubRoutine Gradient9(N1, N2, N3, N2P, Dx, F, V, G)
      Implicit None
      Integer N1, N2, N3, N2P
      Real*8 Dx
      Complex*16 F(N1,N2,N3), V(N1,N3), G(N1,N3)
      
      Integer I, K

C$OMP Parallel Do
C$OMP& Schedule(Static, 1)
C$OMP& Default(Shared)
C$OMP& Private(I, K)
      Do K = 1, N3
         Do I = 1, N1
            V(I, K) = F(I, N2P, K)
            
            G(I,K) = 
     $           - 3*(F(I,N2P+4,K) - F(I,N2P-4,K)) 
     $           + 32*(F(I,N2P+3,K) - F(I,N2P-3,K)) 
     $           - 168*(F(I,N2P+2,K) - F(I,N2P-2,K)) 
     $           + 672*(F(I,N2P+1,K) - F(I,N2P-1,K)) 
            G(I,K) = G(I,K)/(840*Dx)
         EndDo
      EndDo
C$OMP End Parallel Do
      Return
      End


      SubRoutine Gradient11(N1, N2, N3, N2P, Dx, F, V, G)
      Implicit None
      Integer N1, N2, N3, N2P
      Real*8 Dx
      Complex*16 F(N1,N2,N3), V(N1,N3), G(N1,N3)

      Integer I, K

C$OMP Parallel Do
C$OMP& Schedule(Static, 1)
C$OMP& Default(Shared)
C$OMP& Private(I, K)
      Do K = 1, N3
         Do I = 1, N1
            V(I, K) = F(I, N2P, K)
            
            G(I,K) = 
     $           2*(F(I,N2P+5,K) - F(I,N2P-5,K)) 
     $           - 25*(F(I,N2P+4,K) - F(I,N2P-4,K)) 
     $           + 150*(F(I,N2P+3,K) - F(I,N2P-3,K)) 
     $           - 600*(F(I,N2P+2,K) - F(I,N2P-2,K)) 
     $           + 2100*(F(I,N2P+1,K) - F(I,N2P-1,K)) 
            G(I,K) = G(I,K)/(2520*Dx)
         EndDo
      EndDo
C$OMP End Parallel Do
      Return
      End
