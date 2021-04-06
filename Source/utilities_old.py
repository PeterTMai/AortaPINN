"""
@author: Maziar Raissi
"""

import tensorflow as tf
import numpy as np
import pickle

def tf_session():
    # tf session
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.force_gpu_compatible = True
    sess = tf.compat.v1.Session(config=config)
    
    # init
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    
    return sess

def relative_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.sqrt(np.mean(np.square(pred - exact))/np.mean(np.square(exact - np.mean(exact))))
    return tf.sqrt(tf.reduce_mean(input_tensor=tf.square(pred - exact))/tf.reduce_mean(input_tensor=tf.square(exact - tf.reduce_mean(input_tensor=exact))))

def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return tf.reduce_mean(input_tensor=tf.square(pred - exact))

def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    G = tf.gradients(ys=Y, xs=x, grad_ys=dummy)[0]
    Y_x = tf.gradients(ys=G, xs=dummy)[0]
    return Y_x

class neural_net(object):
    def __init__(self, *inputs, layers):
        
        self.layers = layers
        self.num_layers = len(self.layers)
        
        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = X.mean(0, keepdims=True)
            self.X_std = X.std(0, keepdims=True)
        
        self.weights = []
        self.biases = []
        self.gammas = []
        
        for l in range(0,self.num_layers-1):
            in_dim = self.layers[l]
            out_dim = self.layers[l+1]
            W = np.random.normal(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            g = np.ones([1, out_dim])
            # tensorflow variables
            self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True))
            self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
            self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))

    def __call__(self, *inputs):
                
        H = (tf.concat(inputs, 1) - self.X_mean)/self.X_std
    
        for l in range(0, self.num_layers-1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W/tf.norm(tensor=W, axis = 0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g*H + b
            # activation
            if l < self.num_layers-2:
                H = H*tf.sigmoid(H)
                
        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)
    
        return Y


    def save_NN(self, fileDir):

        nn_weights = self.sess.run(self.weights)
        nn_biases = self.sess.run(self.biases)
        nn_gammas = self.sess.run(self.gammas)

        with open(fileDir, 'wb') as f:
            pickle.dump([nn_weights, nn_biases, nn_gammas], f)
            print("Save uv NN parameters successfully...")


    def load_NN(self,fileDir):
        self.weights = []
        self.biases = []
        self.gammas = []

        with open(fileDir, 'rb') as f:
            nn_weights, nn_biases, nn_gammas = pickle.load(f)

            # Stored model must has the same # of layers
            assert self.num_layers-1 == (len(nn_weights))

            for num in range(0, self.num_layers-1):
                W = tf.Variable(nn_weights[num], dtype=tf.float32)
                b = tf.Variable(nn_biases[num], dtype=tf.float32)
                g = tf.Variable(nn_gammas[num], dtype=tf.float32)
                self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True))
                self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
                self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))
                print(" - Load NN parameters successfully...")


def Navier_Stokes_2D(u, v, p, t, x, y, Rey):
    
    Y = tf.concat([u, v, p], 1)
    
    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    
    u = Y[:,0:1]
    v = Y[:,1:2]
    p = Y[:,2:3]
    
    u_t = Y_t[:,0:1]
    v_t = Y_t[:,1:2]
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    p_x = Y_x[:,2:3]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    p_y = Y_y[:,2:3]
    
    u_xx = Y_xx[:,0:1]
    v_xx = Y_xx[:,1:2]
    
    u_yy = Y_yy[:,0:1]
    v_yy = Y_yy[:,1:2]
    
    e1 = u_t + (u*u_x + v*u_y) + p_x - (1.0/Rey)*(u_xx + u_yy) 
    e2 = v_t + (u*v_x + v*v_y) + p_y - (1.0/Rey)*(v_xx + v_yy)
    e3 = u_x + v_y
    
    return e1, e2, e3

def Gradient_Velocity_2D(u, v, x, y):
    
    Y = tf.concat([u, v], 1)
    
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    
    return [u_x, v_x, u_y, v_y]

def Strain_Rate_2D(u, v, x, y):
    
    [u_x, v_x, u_y, v_y] = Gradient_Velocity_2D(u, v, x, y)
    
    eps11dot = u_x
    eps12dot = 0.5*(v_x + u_y)
    eps22dot = v_y
    
    return [eps11dot, eps12dot, eps22dot]

def Navier_Stokes_3D(u, v, w, p, t, x, y, z, Rey):
    
    Y = tf.concat([u, v, w, p], 1)
    
    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_z = fwd_gradients(Y, z)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    Y_zz = fwd_gradients(Y_z, z)
    
    u = Y[:,0:1]
    v = Y[:,1:2]
    w = Y[:,2:3]
    p = Y[:,3:4]
    
    u_t = Y_t[:,0:1]
    v_t = Y_t[:,1:2]
    w_t = Y_t[:,2:3]
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    w_x = Y_x[:,2:3]
    p_x = Y_x[:,3:4]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    w_y = Y_y[:,2:3]
    p_y = Y_y[:,3:4]
       
    u_z = Y_z[:,0:1]
    v_z = Y_z[:,1:2]
    w_z = Y_z[:,2:3]
    p_z = Y_z[:,3:4]
    
    u_xx = Y_xx[:,0:1]
    v_xx = Y_xx[:,1:2]
    w_xx = Y_xx[:,2:3]
    
    u_yy = Y_yy[:,0:1]
    v_yy = Y_yy[:,1:2]
    w_yy = Y_yy[:,2:3]
       
    u_zz = Y_zz[:,0:1]
    v_zz = Y_zz[:,1:2]
    w_zz = Y_zz[:,2:3]
    
    e1 = u_t + (u*u_x + v*u_y + w*u_z) + p_x - (1.0/Rey)*(u_xx + u_yy + u_zz)
    e2 = v_t + (u*v_x + v*v_y + w*v_z) + p_y - (1.0/Rey)*(v_xx + v_yy + v_zz)
    e3 = w_t + (u*w_x + v*w_y + w*w_z) + p_z - (1.0/Rey)*(w_xx + w_yy + w_zz)
    e4 = u_x + v_y + w_z
    
    return e1, e2, e3, e4

def Gradient_Velocity_3D(u, v, w, x, y, z):
    
    Y = tf.concat([u, v, w], 1)
    
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_z = fwd_gradients(Y, z)
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    w_x = Y_x[:,2:3]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    w_y = Y_y[:,2:3]
    
    u_z = Y_z[:,0:1]
    v_z = Y_z[:,1:2]
    w_z = Y_z[:,2:3]
    
    return [u_x, v_x, w_x, u_y, v_y, w_y, u_z, v_z, w_z]

def Shear_Stress_3D(u, v, w, x, y, z, nx, ny, nz, Rey):
        
    [u_x, v_x, w_x, u_y, v_y, w_y, u_z, v_z, w_z] = Gradient_Velocity_3D(u, v, w, x, y, z)

    uu = u_x + u_x
    uv = u_y + v_x
    uw = u_z + w_x
    vv = v_y + v_y
    vw = v_z + w_y
    ww = w_z + w_z
    
    sx = (uu*nx + uv*ny + uw*nz)/Rey
    sy = (uv*nx + vv*ny + vw*nz)/Rey
    sz = (uw*nx + vw*ny + ww*nz)/Rey
    
    return sx, sy, sz
