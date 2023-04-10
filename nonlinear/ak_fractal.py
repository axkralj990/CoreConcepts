import numpy as np
import matplotlib.pyplot as plt

def create_kernel(l, x0, y0, type):
    # Creates the kernel shape.
    #
    # Input: l, x0, y0, type
    # Output: kernel
    #
    # type 0: Koch curve
    # type 1: zigzag
    # type 2: square
    #
    # Written by: Aleksij Kraljic
    
    if (type == 1):
        kernel = np.array([[[x0, y0], [x0+l, y0]], [[x0+l, y0], [x0+l, y0+l]], [[x0+l, y0+l], [x0+l+l, y0+l]]])
    elif (type == 2):
        kernel = np.array([[[x0, y0], [x0+l, y0]], [[x0+l, y0], [x0+l, y0+l]], [[x0+l, y0+l], [x0+l+l, y0+l]], [[x0+l+l, y0+l], [x0+l+l, y0]], [[x0+l+l, y0], [x0+l+l+l, y0]]])
    else:
        kernel = np.array([[[x0, y0], [x0+l, y0]], [[x0+l, y0], [x0+l+l/2, y0+np.sqrt(3)/2*l]], [[x0+l+l/2, y0+np.sqrt(3)/2*l], [x0+l+l, y0]], [[x0+l+l, y0], [x0+l+l+l, y0]]])
    return kernel

def plot_curve(curve):
    # Plots the curve.
    #
    # Input: curve
    #
    # Written by: Aleksij Kraljic
    
    for segment in curve:
        plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 'r')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def rotate_segments(curve, theta):
    # Rotates the segments by theta degrees.
    #
    # Input: curve, theta
    # Output: curve
    #
    # Written by: Aleksij Kraljic
    
    R = np.matrix([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)]).reshape(2,2)
    for i in range(0,len(curve)):
        curve[i][0] = np.matmul(R, curve[i][0]).tolist()[0]
        curve[i][1] = np.matmul(R, curve[i][1]).tolist()[0]
    return curve

def translate_segments(curve, x0, y0):
    # Translates the segments by x0 and y0.
    #
    # Input: curve, x0, y0
    # Output: curve
    #
    # Written by: Aleksij Kraljic
    
    for i in range(0,len(curve)):
        curve[i][0][0] += x0
        curve[i][0][1] += y0
        curve[i][1][0] += x0
        curve[i][1][1] += y0
    return curve

def prepare_kernel(curve):
    # Prepares the kernel shape by
    # translating it to origin and rotating it to x-axis
    #
    # Input: curve
    # Output: kernel
    #
    # Written by: Aleksij Kraljic
    
    kernel = np.copy(curve)
    x0 = kernel[0][0][0]
    y0 = kernel[0][0][1]
    x1 = kernel[0][1][0]
    y1 = kernel[0][1][1]
    translate_segments(kernel, -x0, -y0)
    a = np.array([x0,y0])
    b = np.array([x1,y1])
    c = b - a
    theta = np.arctan2(c[1],c[0])*180/np.pi
    rotate_segments(kernel, -theta)
    return kernel

def iterate_fractal(curve, kernel, N):
    # Iterates the fractal curve
    #
    # Input: curve, kernel, N
    # Output: curve
    #
    # Written by: Aleksij Kraljic
    path_element_length = np.sqrt((curve[0][1][0] - curve[0][0][0])**2 + (curve[0][1][1] - curve[0][0][1])**2)
    kernel_length = kernel[-1][1][0]-kernel[0][0][0]
    kernel = kernel * path_element_length/kernel_length
    if (N == 0):
        return(curve)
    else:
        new_curve = np.empty((len(curve)*(len(kernel)), 2, 2))
        pXt = curve[0][0][0]
        pYt = curve[0][0][1]
        for i in range(0,len(curve)):
            segment = np.copy(kernel)
            x0 = curve[i][0][0]
            y0 = curve[i][0][1]
            x1 = curve[i][1][0]
            y1 = curve[i][1][1]
            a = np.array([x0,y0])
            b = np.array([x1,y1])
            c = b - a
            theta = np.arctan2(c[1],c[0])
            rotate_segments(segment, theta)
            translate_segments(segment, pXt, pYt)
            new_curve[len(segment)*i:len(segment)+i*len(segment)][:][:] = segment
            pXt = segment[-1][1][0]
            pYt = segment[-1][1][1]
        return iterate_fractal(new_curve, kernel, N-1)

def create_path(x0, y0, l, type):
    # Creates a path for the fractal curve
    #
    # Input: x0, y0, l, type
    # Output: path
    #
    # type = 0: line
    # type = 1: square
    # type = 2: triangle
    #
    # Written by: Aleksij Kraljic
    
    if (type == 1):
        path = np.array([[[x0,y0],[x0+l/2,y0+np.sqrt(3)/2*l]], [[x0+l/2,y0+np.sqrt(3)/2*l],[l+x0,y0]], [[l+x0,y0],[x0,y0]]])
    elif (type == 2):
        path = np.array([[[x0,y0],[x0,y0+l]], [[x0,y0+l],[x0+l,y0+l]], [[x0+l,y0+l],[x0+l,y0]], [[x0+l,y0],[x0,y0]]])
    else:
        path = np.array([[[x0,y0],[x0+l,y0]]])
    return path

if __name__ == "__main__":
    l = 4
    x0 = 0
    y0 = 0
    path = create_path(x0, y0, l, 0)
    kernel = create_kernel(l, x0, y0, 2)
    kernel = prepare_kernel(kernel)
    curve = iterate_fractal(path, kernel, 3)
    plot_curve(curve)
