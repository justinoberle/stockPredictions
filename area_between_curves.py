# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 17:21:32 2021

@author: justinoberle
"""
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

x = [1,2,3,4,5,6,7,8,9,10]
y1 = [1,2,3,4,5,6,7,8,9,10]
y2 = [2,1,2,4,1,5,7,3,12,15]

def best_fit_slope_and_intercept(xs,ys):
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))
    b = np.mean(ys) - m*np.mean(xs)
    return m, b

# fig, axs = plt.subplots(2, 1, figsize=(10,4))
# axs[0].plot(x, y1, label='y1')
# axs[0].plot(x, y2, label='y2')
# create segments and boundaries at start and end for each segment. a segment is [x1, x2] of a much larger x array
def get_area_enclosed_by_2_curves(x, y1, y2):
    #x, y1, y2 must all be same length
    area_total_list = []
    for i in range(len(x) - 1): 
        a = i
        b = a+1
        new_x = np.array([x[a], x[b]])
        new_y1 = np.array([y1[a], y1[b]])
        new_y2 = np.array([y2[a], y2[b]])
        # bound_begin_x = [x[a], x[a]]
        # bound_begin_y = [y1[a], y2[a]]
        # bound_end_x = [x[b], x[b]]
        # bound_end_y = [y1[b], y2[b]]
        # find equation of each line segment
        y1_m, y1_b = best_fit_slope_and_intercept(new_x, new_y1)
        y2_m, y2_b = best_fit_slope_and_intercept(new_x, new_y2)
        # y1_equation = y1_m*new_x + y1_b
        # y2_equation = y2_m*new_x + y2_b
        # find intersection which is when y1 = y2 solve for x_intersection
        if y2_m - y1_m == 0:
            # this will be inf because it does not intersect. In this case use new_x[1] as x_intercept
            x_intersection = new_x[1]
        else:
            x_intersection = (y1_b - y2_b) / (y2_m - y1_m)
        if new_x[0] == x_intersection or (x_intersection <= new_x[0] or x_intersection >= new_x[1]):
            x_intersection = new_x[1]
        y1_integration_equation = lambda x: (y1_m*x + y1_b)
        y2_integration_equation = lambda x: (y2_m*x + y2_b)
        area_y1 = integrate.quad(y1_integration_equation, new_x[0], x_intersection)
        area_y2 = integrate.quad(y2_integration_equation, new_x[0], x_intersection)
        area = np.abs(area_y1[0] - area_y2[0])
        
        if x_intersection == new_x[0] or x_intersection == new_x[1]:
            # intersects but at edge or outside of range
            # print("intersect at edge")
            if new_x[0] == x_intersection:
                x_intersection = new_x[1]
        else:
            # intersection in middle somewhere
            # print("middle intersection")
            # intersection_x_plotting = np.array([x_intersection, x_intersection])
            # y = y1_m*x_intersection + y1_b
            # intersection_y = np.array([y-1, y+1])
            # axs[1].plot(intersection_x_plotting, intersection_y, '--')
            area_y1_2 = integrate.quad(y1_integration_equation, x_intersection, new_x[1])
            area_y2_2 = integrate.quad(y2_integration_equation, x_intersection, new_x[1])
            area = area + np.abs(area_y1_2[0] - area_y2_2[0])
        area_total_list.append(area)
        
    area_total = sum(area_total_list)
    return area_total



# axs[0].plot(bound_begin_x, bound_begin_y, label='beginning bound')
# axs[0].plot(bound_end_x, bound_end_y, label='y2')
# # axs[1].plot(new_x, new_y2, label='y2 small segment')
# axs[1].plot(new_x, y1_equation, label='y1 equation')
# axs[1].plot(new_x, y2_equation, label='y2 equation')
# axs[1].plot(bound_begin_x, bound_begin_y, label='beginning bound')
# axs[1].plot(bound_end_x, bound_end_y, label='y2')
# axs[0].legend()
# axs[1].legend()
# fig.tight_layout()
# plt.show()

