import numpy as np
import matplotlib.pyplot as plt
import data_parser as dp
import gradient_descent as gd
import time

points = dp.get_data()

def compute_total_error(m,b): #Computes total mean squared error
    totalError = 0
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y - (m * x + b)) ** 2 #Error is calculated as y' = mx + b(Assuming linear regression) so E = (y-y')^2, summed over all points

    return totalError/float(len(points)) #Returning the mean squared error.

def total_error(point_pair): #driver function for compute_total_error
    return compute_total_error(point_pair[0], point_pair[1])

def compute_jacobian(point_pair, h = 1e-5): #computes the jacobian of the function total_error
    n = len(point_pair)
    jacobian = np.zeros(n) #initialize the jacobian matrix
    for i in range(n):
        x_i = np.zeros(n)
        x_i[i] += h #add the limit value, any small value > 0 should do
        jacobian[i] = (total_error(point_pair+x_i) - total_error(point_pair))/h #calculate derivative using first principle method f'(x) = lt(h->0) (f(x+h) - f(x))/h
    return jacobian #return the jacobian for the pair of points

def compute_hessian(point_pair, h = 1e-5): #computes the hessian of the function total_error, it is found as the derivative of the jacobian
    n = len(point_pair)
    hessian = np.zeros((n,n)) #initialize the hessian matrix
    for i in range(n):
        x_i = np.zeros(n)
        x_i[i] += h #add the limit value, any small value > 0 should do
        hessian[i] = (compute_jacobian(point_pair+x_i) - compute_jacobian(point_pair))/h #calculate derivative using first principle method f'(x) = lt(h->0) (f(x+h) - f(x))/h

    return hessian #return the jacobian for the pair of points

def compute_newton(init_points, max_iter = 10000, e = 1e-5): #calculate roots of the equation, i.e. find x if f(x) = 0. In our case we want to find the minima point, so we find f'(x) = 0
    point_pair_arr = np.zeros((max_iter, len(init_points))) #initalize m,b values
    point_pair_arr[0] = init_points #start points
    opt_val = None #optimal_value to return
    for i in range(max_iter):
        jacobian = compute_jacobian(point_pair_arr[i]) #calculate the jacobian at current m,b
        hessian = compute_hessian(point_pair_arr[i]) #calculate the hessian at current m,b
        point_pair_arr[i+1] = point_pair_arr[i] - np.dot(np.linalg.pinv(hessian), jacobian) #calulate the new m, new b using newton's equation x(t+1) = x(t) - f(x(t))/f'(x(t)) but we want to find root of f'(x) so we would do x(t+1) = x(t) - f'(x(t))/f''(x(t))
        #pinv is pseudo inverse, it prevents values like 1/0 and replaces it with a very high value.
        print('New m is %.2f and new b is %.2f'%(point_pair_arr[i,0], point_pair_arr[i,1]))
        opt_val = point_pair_arr[i+1]
        if np.abs(total_error(point_pair_arr[i+1]) - total_error(point_pair_arr[i])) < e: #used for early stopping, stops when there is no real improvement.
            print('Optimal m is %.2f and Optimal b is %.2f'%(point_pair_arr[i+1,0], point_pair_arr[i+1,1]))
            break

    return opt_val

def plot_line_data(m, b): #Plots the calculated line from m and b
    X = points[:,0]
    Y = points[:,1]
    plt.plot(X, Y, 'bo') #First plots the data points
    plt.plot(X, m * X + b) #Plot the line.
    plt.axis([0,1.5* max(X), 0, 1.3 * max(Y)]) #Set the axes range.
    plt.title("Best line.")
    plt.text(10, 130, "m="+str(round(m,4))+"  b="+str(round(b,4)) + " error="+str(compute_total_error(m,b)))
    plt.show() #shows the graph.
    return

def main(): #main driver function
    init_points = np.array([0.0,1.0]) #intial points
    print("2nd order optimization starts at "+ str(time.asctime())) #start time
    time_t = time.time() #start time
    newton_points = compute_newton(init_points, max_iter = 100) #find the solution
    print(newton_points)
    print("b = {0}, m = {1}, error = {2}".format(newton_points[1], newton_points[0], compute_total_error(newton_points[0], newton_points[1])))
    time_t = time.time() - time_t #end time
    print("2nd order optimization ends at %s and has taken %dms"%(str(time.asctime()), time_t))
    plot_line_data(newton_points[0], newton_points[1]) #plot the line generated
    print("1st order optimization starts at "+ str(time.asctime())) #start time
    time_t = time.time()
    m,b = gd.run()
    time_t = time.time() - time_t #end time
    print("1st order optimization ends at %s and has taken %dms"%(str(time.asctime()), time_t))
    plot_line_data(m, b) #plot the generated line
    return

if __name__=='__main__':
    main()
