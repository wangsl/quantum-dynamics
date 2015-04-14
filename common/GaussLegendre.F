
C     $Id$

      SubRoutine GaussLegendre(N, Points, Weight)
      Implicit None
      Integer N
      Real*8 Points(N), Weight(N)
      
      Integer NMax
      Parameter(NMax = 199)
      
      Integer I, J, IStart
      
#include "gauss-legendre-199.inc"
      
      IStart = Idx(N)
      
      J = 1
      If(N/2*2 .ne. N) then
         Points(1) = X(IStart)
         Weight(1) = W(IStart)
         IStart = IStart + 1
         J = J + 1
      EndIf
      
      Do I = 1, N/2
         Points(J) = X(IStart+I-1)
         Weight(J) = W(IStart+I-1)
         J = J+1
         Points(J) = -X(IStart+I-1)
         Weight(J) = W(IStart+I-1)
         J = J + 1
      EndDo

      Call Sort2Double(Points, Weight, N)

      Return
      End

