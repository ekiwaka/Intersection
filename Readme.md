# Intersection

Assignment details can be seen in the assignment pdf.

Dependencies:
  numpy
  cv2
  sympy

Usage:
  Change name of the image to be tested in the main function. The program returns the x and y coordinates of the point of intersection, or returns a "no intersection" message.
  

Caveats:
  The program fails if the line segment is very short, since the Canny edge detection fails to give enough weightage to very small line segments.
  The circle and line detectors have slight drifts, which can be rectified by:-
    a) Further fine tuning the detectors
    b) Make a network capable to learn and adjust 
