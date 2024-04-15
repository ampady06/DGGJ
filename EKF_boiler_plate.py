

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open(r'C:\Users\arjun\OneDrive\Desktop\EKF\data\data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']  # timestamps [s]

x_init  = data['x_init'] # initial x position [m]
y_init  = data['y_init'] # initial y position [m]
th_init = data['th_init'] # initial theta position [rad]

# input signal
v  = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]


## Testing the Unpacked Data
x_init.shape
y_init.shape
th_init.shape

print(v.shape)
print(len(v))

om.shape
b.shape
r.shape
l.shape

print(d.shape)
print(d[0])


## Initializing Parameters

v_var = 0.01  # translation velocity variance
om_var = 0.01  # rotational velocity variance
# allowed to tune these values
# r_var = 0.1  # range measurements variance
r_var = 0.01
# b_var = 0.1  # bearing measurement variance
b_var = 10

Q_km = np.diag([v_var, om_var]) # input noise covariance
cov_y = np.diag([r_var, b_var])  # measurement noise covariance

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init]) # initial state
P_est[0] = np.diag([1, 1, 0.1]) # initial state covariance


'''Now that our data is loaded, we can begin getting things set up for our solver. 
   One of the most important aspects of designing a filter is determining the input and measurement noise covariance matrices, 
   as well as the initial state and covariance values. We set the values here: '''

print(Q_km.shape)
print(Q_km)

print(cov_y.shape)
print(cov_y)

x_est.shape

print(P_est.shape)
print(P_est[0])
print(P_est[1])


'''Remember: that it is neccessary to tune the measurement noise variances r_var, b_var in order for the filter to perform well!
   In order for the orientation estimates to coincide with the bearing measurements, it is also neccessary to wrap all estimated 
   θ  values to the  (−π,π]  range.'''

# Wraps angle to (-pi,pi] range
def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x



## Correction Step

'''First, implement the measurement update function, 
   which takes an available landmark measurement  l  and updates the current state estimate  xˇk . 
   For each landmark measurement received at a given timestep  k , you should implement the following steps given in corrector.png '''

def measurement_update(lk, rk, bk, P_check, x_check):
    x_k = x_check[0]
    y_k = x_check[1]
    theta_k = wraptopi(x_check[2])

    x_l = lk[0]
    y_l = lk[1]

    d_x = x_l - x_k - d*np.cos(theta_k)
    d_y = y_l - y_k - d*np.sin(theta_k)

    r = np.sqrt(d_x**2 + d_y**2)
    phi = np.arctan2(d_y, d_x) - theta_k

    # 1. Compute measurement Jacobian
    H_k = np.zeros((2,3))
    H_k[0,0] = -d_x/r
    H_k[0,1] = -d_y/r
    H_k[0,2] = d*(d_x*np.sin(theta_k) - d_y*np.cos(theta_k))/r
    H_k[1,0] = d_y/r**2
    H_k[1,1] = -d_x/r**2
    H_k[1,2] = -1-d*(d_y*np.sin(theta_k) + d_x*np.cos(theta_k))/r**2

    M_k = np.identity(2)

    y_out = np.vstack([r, wraptopi(phi)])
    y_mes = np.vstack([rk, wraptopi(bk)])

    # 2. Compute Kalman Gain
    K_k = P_check.dot(H_k.T).dot(np.linalg.inv(H_k.dot(P_check).dot(H_k.T) + M_k.dot(cov_y).dot(M_k.T)))

    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])
    x_check = x_check + K_k.dot(y_mes - y_out)
    x_check[2] = wraptopi(x_check[2])

    # 4. Correct covariance
    P_check = (np.identity(3) - K_k.dot(H_k)).dot(P_check)

    return x_check, P_check



## Prediction Step : 

'''Now, implement the main filter loop, defining the prediction step of the EKF using the motion model provided '''

'''HINT : hint.png '''

#### 5. Main Filter Loop #######################################################################
# set the initial values
P_check = P_est[0]
x_check = x_est[0, :].reshape(3,1)
for k in range(1, len(t)):  # start at 1 because we've set the initial prediciton

    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)
    theta = wraptopi(x_check[2])

    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])
#     x_check = np.zeros(3)
    F = np.array([[np.cos(theta), 0, -np.sin(theta)*delta_t*v[k-1]],
              [np.sin(theta), 0, np.cos(theta)*delta_t*v[k-1]],
              [0, 1, 0]], dtype='float')
    inp = np.array([[v[k-1]], [om[k-1]], [0]])

    x_check = x_check + np.dot(F, inp) * delta_t
    x_check[2] = wraptopi(x_check[2])

    # 2. Motion model jacobian with respect to last state
    F_km = np.zeros([3, 3])
    F_km = np.array([[1, 0, -np.sin(theta)*delta_t*v[k-1]],
                 [0, 1, np.cos(theta)*delta_t*v[k-1]],
                 [0, 0, 1]], dtype='float')
    # dtype='float'

    # 3. Motion model jacobian with respect to noise
    L_km = np.zeros([3, 2])
    L_km = np.array([[np.cos(theta)*delta_t, 0], 
                 [np.sin(theta)*delta_t, 0],
                 [0, delta_t]], dtype='float')

    # 4. Propagate uncertainty
    P_check = F_km.dot(P_check.dot(F_km.T)) + L_km.dot(Q_km.dot(L_km.T)) 

     # 5. Update state estimate using available landmark measurements
    for i in range(len(r[k])):
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)

    # Set final state predictions for timestep
    x_est[k, 0] = x_check[0]
    x_est[k, 1] = x_check[1]
    x_est[k, 2] = x_check[2]
    P_est[k, :, :] = P_check



## Plot the resulting state estimates:
e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(x_est[:, 0], x_est[:, 1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.show()
plt.close()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], x_est[:, 2])
ax.set_xlabel('Time [s]')
ax.set_ylabel('theta [rad]')
ax.set_title('Estimated trajectory')
plt.show()
plt.close() 


with open(r'C:\Users\arjun\OneDrive\Desktop\EKF\submission.pkl', 'wb') as f:
    pickle.dump(x_est, f, pickle.HIGHEST_PROTOCOL)
