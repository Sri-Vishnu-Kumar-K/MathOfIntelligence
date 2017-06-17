import data_parser as dp
import matplotlib.pylab as plt

def compute_total_error(b,m,points):
    totalError = 0
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y - (m * x + b)) ** 2

    return totalError/float(len(points))

def step_gradient(curr_m, curr_b, points, lr):
    dt_db = 0
    dt_dm = 0
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        t = y - (curr_m*x + curr_b)
        dt_dm += -1 * x * t
        dt_db += -1 * t

    dt_dm = (2*dt_dm)/float(len(points))
    dt_db = (2*dt_db)/float(len(points))

    m = curr_m - (lr * dt_dm)
    b = curr_b - (lr * dt_db)

    return [m,b]

def gradient_descent_driver(points, start_m, start_b, lr, num_iterations, early_stop_number = 0, modify_lr = False):
    m = start_m
    b = start_b
    stop_number = 0
    error = []
    modify_lr_num = 3
    er_decrease = 0
    for i in range(num_iterations):
        m, b = step_gradient(m, b, points, lr)
        er = compute_total_error(b,m,points)
        er = round(er,2)
        # print(er)
        if i>=1 and error[len(error)-1] == er and early_stop_number != 0:
            stop_number += 1
            if stop_number == early_stop_number:
                print('Executing early stopping')
                break

        if modify_lr and er<compute_total_error(b,m,points):
            er_decrease+=1
            if er_decrease%modify_lr_num == 0:
                print('Increasing lr for faster descent')
                lr += 0.00002
        error.append(er)

    print('Finished with an lr of %f'%(lr))
    return [m,b,error]

def plot_line_data(points, m, b):
    X = points[:,0]
    Y = points[:,1]
    plt.plot(X, Y, 'bo')
    plt.plot(X, m * X + b)
    plt.axis([0,1.5* max(X), 0, 1.3 * max(Y)])
    plt.title("Best fit : Linear Regression")
    plt.text(10, 130, "m="+str(round(m,4))+"  b="+str(round(b,4)))
    plt.show()
    return

def plot_error_data(error_value):
    num_iterations = range(len(error_value))
    print('min_error and max_errors are %.2f, %.2f'%(min(error_value), max(error_value)))
    plt.plot(num_iterations, error_value)
    plt.axis([0,1.5*max(num_iterations), min(error_value)-5, max(error_value) + 5])
    plt.xlabel('iterations')
    plt.ylabel('error_value')
    plt.text(10,130,'Min_Error value is :'+str(min(error_value)))
    plt.show()
    return

def run():
    points = dp.get_data()
    learning_rate = 0.00005
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_total_error(initial_b, initial_m, points)))
    print("Running...")
    [m, b, error] = gradient_descent_driver(points, initial_b, initial_m, learning_rate, num_iterations, early_stop_number = 5, modify_lr = True)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_total_error(b, m, points)))
    plot_line_data(points, m, b)
    plot_error_data(error)
    return

if __name__ == '__main__':
    run()
