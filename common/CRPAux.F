
C     $Id$

      SubRoutine PsiTimeToPsiEnergyOnSurface(
     $     N1, NTheta, NE,
     $     ExpIETDt, PsiT, DPsiT, faiE, DFaiE
     $     )
      Implicit None
      Integer  N1, NTheta, NE
      Complex*16 ExpIETDt(NE)
      Complex*16 PsiT(N1,NTheta), DPsiT(N1,NTheta)
      Complex*16 faiE(N1,NTheta,NE), DfaiE(N1,NTheta,NE)
      
      Integer IE, I1, I3
      
C$OMP Parallel Do
C$OMP& Schedule(Static, 1)
C$OMP& Default(Shared)
C$OMP& Private(I1, I3, IE)
      Do IE = 1, NE
         Do I3 = 1, NTheta
            Do I1 = 1, N1
               faiE(I1,I3,IE) = faiE(I1,I3,IE) + ExpIETDt(IE)*PsiT(I1,I3)
               DfaiE(I1,I3,IE) = DfaiE(I1,I3,IE) + ExpIETDt(IE)*DPsiT(I1,I3)
            EndDo
         EndDo
      EndDo
C$OMP End Parallel Do

      Return
      End


      SubRoutine CalculateCRP(N1, NTheta, NE, Dr1, mu2, W, EtaSq,
     $     Fai, DFai, CRP)
      Implicit None
      Integer N1, NTheta, NE
      Real*8 Dr1, mu2
      Complex*16 Fai(N1,NTheta,NE), DFai(N1,NTheta,NE)
      Real*8 W(NTheta), EtaSq(NE), CRP(NE)

      Integer IE, I1, I3
      Complex*16 CE, C3
      Real*8 Dr1Mu2

      Dr1Mu2 = Dr1/mu2

C$OMP Parallel Do
C$OMP& Schedule(Static, 1)
C$OMP& Default(Shared)
C$OMP& Private(I1, I3, IE, CE, C3)
      Do IE = 1, NE
         CE = (0.0D0, 0.0D0)
         Do I3 = 1, NTheta
            C3 = (0.0D0, 0.0D0)
            Do I1 = 1, N1
               C3 = C3 + Conjg(Fai(I1,I3,IE))*DFai(I1,I3,IE)
            EndDo
            CE = CE + W(I3)*C3
         EndDo
         CRP(IE) = Imag(CE)/EtaSq(IE)*Dr1Mu2
      EndDo
C$OMP End Parallel Do

      Return
      End
