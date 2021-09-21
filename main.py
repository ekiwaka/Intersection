"""
Program to detect the point of intersection of a circular arc and a line segment
in a given image

Author: Karan Rawat
"""

import numpy as np
import cv2
from sympy import Circle, Point, Line, intersection


def detect_circle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.85, 65,
                               param1=60, param2=30, minRadius=0, maxRadius=0)

    radius_map = {}
    for n in range(0, 500, 1):
        radius_map[n] = []

    print(circles)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(circles)

        for (x, y, r) in circles:
            radius_map[r].append((x, y, r))

        radius_approx = circles[0][2]
        print(radius_approx)
        circle_approx = radius_map[radius_approx]
        output = gray.copy()
        for x, y, r in circle_approx:
            cv2.circle(output, (x, y), r, (0, 0, 255), 1)
            cv2.imshow(f"Radius {radius_approx}", output)

        return circle_approx[0]


def euler_to_coordinate_transform(euler_line):
    for rho, theta in euler_line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * a)
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * a)
    return [(x1, y1), (x2, y2)]


def detect_line(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 60, apertureSize=3)

    cv2.imshow("edges", edges)
    lines = cv2.HoughLines(edges, 1.85, np.pi / 360, 90)

    print (lines)
    if lines is not None:
        line_transform = euler_to_coordinate_transform(lines[0])
        cv2.line(img, line_transform[0], line_transform[1], (255, 0, 0), 1)
        cv2.imshow("Line", img)

    return [line_transform[0], line_transform[1]]


if __name__ == '__main__':

    original_image = cv2.imread('ex3.png')
    cv2.imshow('Original image', original_image)
    cv2.waitKey(0)

    detected_circle = detect_circle(original_image)
    detected_line = detect_line(original_image)

    line = Line(Point(detected_line[0]), Point(detected_line[1]))
    circle = Circle(Point(detected_circle[0], detected_circle[1]), detected_circle[2])
    result = intersection(circle, line)

    if result == []:
        print("No intersection between the circle and the line in the given image")

    else:
        print("The circle and the line intersect at : ",
              "(", float(result[0].x), ",", float(result[0].y), ")",
              "(", float(result[1].x), ",", float(result[1].y), ")")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
