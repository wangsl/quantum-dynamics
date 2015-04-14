
C     $Id$

C     Double and complex matrix multiplication

      SubRoutine ForwardLegendreTransform(
     $     CPsi, LPsi, NR, NTheta, NLeg, WLegP)
      Implicit None
      Integer NR, NTheta, NLeg
      Complex*16 CPsi(NR, NTheta), LPsi(NR, NLeg)
      Real*8 WLegP(NTheta, NLeg)

      Integer I, K, L

      Call DCZeros(NR*NLeg, LPsi)
      
C$OMP Parallel Do
C$OMP& Schedule(Static, 1)
C$OMP& Default(Shared)
C$OMP& Private(L, K, I)
      Do L = 1, NLeg
         Do K = 1, NTheta
            Do I = 1, NR
               LPsi(I,L) = LPsi(I,L) + CPsi(I,K)*WLegP(K,L)
            EndDo
         EndDo
      EndDo
C$OMP End Parallel Do
      
      Return
      End


      SubRoutine BackwardLegendreTransform(
     $     CPsi, LPsi, NR, NTheta, NLeg, LegP)
      Implicit None
      Integer NR, NTheta, NLeg
      Complex*16 CPsi(NR, NTheta), LPsi(NR, NLeg)
      Real*8 LegP(NLeg, NTheta)
      
      Integer I, K, L

      Call DCZeros(NR*NTheta, CPsi)

C$OMP Parallel Do
C$OMP& Schedule(Static, 1)
C$OMP& Default(Shared)
C$OMP& Private(L, K, I)
      Do K = 1, NTheta
         Do L = 1, NLeg
            Do I = 1, NR
               CPsi(I,K) = CPsi(I,K) + LPsi(I,L)*LegP(L,K)
            EndDo
         EndDo
      EndDo
C$OMP End Parallel Do

      Return
      End
      
