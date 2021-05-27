"""
@author: Thanh Trung Mai
"""

import tensorflow as tf
import numpy as np
import pickle
import time
import sys


class HFM(object):
    # notational conventions
    # _tf: placeholders for input/output data andpoints used to regress the equations
    # _pred: output of neural network
    # _eqns: points used to regress the equations
    # _data: input-output data
    # _star: preditions

    def __init__(self, t_data, x_data, y_data, z_data,
                       t_eqns, x_eqns, y_eqns, z_eqns,
                       u_data, v_data, w_data,
                       layers, batch_size, Rey, ExistModel=0, uvDir=''):

        # specs
        self.layers = layers
        self.batch_size = batch_size

        # flow properties
        self.Rey = Rey

        # data
        [self.t_data, self.x_data, self.y_data, self.z_data, 
         self.u_data, self.v_data, self.w_data] = [t_data, x_data, y_data, z_data, u_data, v_data, w_data]
        [self.t_eqns, self.x_eqns, self.y_eqns, self.z_eqns] = [t_eqns, x_eqns, y_eqns, z_eqns]

        # placeholders
        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.z_data_tf,
         self.u_data_tf, self.v_data_tf, self.w_data_tf] = [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(7)]
        [self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf, self.z_eqns_tf] = [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
        [self.nx_eqns_tf, self.ny_eqns_tf, self.nz_eqns_tf] = [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]


        self.net_cuvwp = neural_net(self.t_data, self.x_data, self.y_data, self.z_data, layers = self.layers)

        if ExistModel==0:
            self.net_cuvwp = neural_net(self.t_data, self.x_data, self.y_data, self.z_data, layers = self.layers)
        else:
            self.net_cuvwp = neural_net(self.t_data, self.x_data, self.y_data, self.z_data, layers = self.layers, load=1, fileDir=uvDir)

        [self.u_data_pred,
         self.v_data_pred,
         self.w_data_pred, self.p_data_pred] = self.net_cuvwp(self.t_data_tf,
                                                              self.x_data_tf,
                                                              self.y_data_tf,
                                                              self.z_data_tf)

        # physics "informed" neural networks
        [self.u_eqns_pred,
         self.v_eqns_pred,
         self.w_eqns_pred,
         self.p_eqns_pred] = self.net_cuvwp(self.t_eqns_tf,
                                            self.x_eqns_tf,
                                            self.y_eqns_tf,
                                            self.z_eqns_tf)

        [self.e1_eqns_pred,
         self.e2_eqns_pred,
         self.e3_eqns_pred,
         self.e4_eqns_pred] = Navier_Stokes_3D(self.u_eqns_pred,
                                               self.v_eqns_pred,
                                               self.w_eqns_pred,
                                               self.p_eqns_pred,
                                               self.t_eqns_tf,
                                               self.x_eqns_tf,
                                               self.y_eqns_tf,
                                               self.z_eqns_tf,
                                               self.Rey)

        [self.sx_eqns_pred,
         self.sy_eqns_pred,
         self.sz_eqns_pred] = Shear_Stress_3D(self.u_eqns_pred,
                                              self.v_eqns_pred,
                                              self.w_eqns_pred,
                                              self.x_eqns_tf,
                                              self.y_eqns_tf,
                                              self.z_eqns_tf,
                                              self.nx_eqns_tf,
                                              self.ny_eqns_tf,
                                              self.nz_eqns_tf,
                                              self.Rey)

        # loss
        self.loss = (mean_squared_error(self.u_data_pred, self.u_data_tf) + \
                    mean_squared_error(self.v_data_pred, self.v_data_tf) + \
                    mean_squared_error(self.w_data_pred, self.w_data_tf))*100 + \
                    mean_squared_error(self.e1_eqns_pred, 0.0) + \
                    mean_squared_error(self.e2_eqns_pred, 0.0) + \
                    mean_squared_error(self.e3_eqns_pred, 0.0) + \
                    mean_squared_error(self.e4_eqns_pred, 0.0)

        # optimizers
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

        self.sess = tf_session()


    def train(self, total_time, learning_rate):

        N_data = self.t_data.shape[0]
        N_eqns = self.t_eqns.shape[0]
           
        start_time = time.time()
        running_time = 0
        it = 0
        while running_time < total_time:
                
            idx_data = np.random.choice(N_data, self.batch_size)
            idx_eqns = np.random.choice(N_eqns, self.batch_size)

            (t_data_batch,
             x_data_batch,
             y_data_batch,
             z_data_batch,
             u_data_batch,
             v_data_batch,
             w_data_batch) = (self.t_data[idx_data,:],
                              self.x_data[idx_data,:],
                              self.y_data[idx_data,:],
                              self.z_data[idx_data,:],
                              self.u_data[idx_data,:],
                              self.v_data[idx_data,:],
                              self.w_data[idx_data,:])

            (t_eqns_batch,
             x_eqns_batch,
             y_eqns_batch,
             z_eqns_batch) = (self.t_eqns[idx_eqns,:],
                              self.x_eqns[idx_eqns,:],
                              self.y_eqns[idx_eqns,:],
                              self.z_eqns[idx_eqns,:])


            tf_dict = {self.t_data_tf: t_data_batch,
                       self.x_data_tf: x_data_batch,
                       self.y_data_tf: y_data_batch,
                       self.z_data_tf: z_data_batch,
                       self.u_data_tf: u_data_batch,
                       self.v_data_tf: v_data_batch,
                       self.w_data_tf: w_data_batch,
                       self.t_eqns_tf: t_eqns_batch,
                       self.x_eqns_tf: x_eqns_batch,
                       self.y_eqns_tf: y_eqns_batch,
                       self.z_eqns_tf: z_eqns_batch,
                       self.learning_rate: learning_rate}

            self.sess.run([self.train_op], tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed/3600.0
                [loss_value,
                 learning_rate_value] = self.sess.run([self.loss,
                                                       self.learning_rate], tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
                      %(it, loss_value, elapsed, running_time, learning_rate_value))
                sys.stdout.flush()
                start_time = time.time()
            it += 1

    def predict(self, t_star, x_star, y_star, z_star):

        tf_dict = {self.t_data_tf: t_star, self.x_data_tf: x_star, self.y_data_tf: y_star, self.z_data_tf: z_star}

        u_star = self.sess.run(self.u_data_pred, tf_dict)
        v_star = self.sess.run(self.v_data_pred, tf_dict)
        w_star = self.sess.run(self.w_data_pred, tf_dict)
        p_star = self.sess.run(self.p_data_pred, tf_dict)

        return u_star, v_star, w_star, p_star


    def save_NN(self, fileDir):

        nn_weights = self.sess.run(self.net_cuvwp.weights)
        nn_biases = self.sess.run(self.net_cuvwp.biases)
        nn_gammas = self.sess.run(self.net_cuvwp.gammas)

        with open(fileDir, 'wb') as f:
            pickle.dump([nn_weights, nn_biases, nn_gammas], f)
            print("Save uv NN parameters successfully...")


    def predict_shear(self, t_star, x_star, y_star, z_star, nx_star, ny_star, nz_star):
        
        tf_dict = {self.t_eqns_tf: t_star, self.x_eqns_tf: x_star, self.y_eqns_tf: y_star, self.z_eqns_tf: z_star,
                   self.nx_eqns_tf: nx_star, self.ny_eqns_tf: ny_star, self.nz_eqns_tf: nz_star}
        
        sx_star = self.sess.run(self.sx_eqns_pred, tf_dict)
        sy_star = self.sess.run(self.sy_eqns_pred, tf_dict)
        sz_star = self.sess.run(self.sz_eqns_pred, tf_dict)
        
        return sx_star, sy_star, sz_star


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
    def __init__(self, *inputs, layers, load=0, fileDir=''):
        
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
        
        if load==0:
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
        else:
            with open(fileDir, 'rb') as f:
                nn_weights, nn_biases, nn_gammas = pickle.load(f)

                # Stored model must has the same # of layers
                assert self.num_layers-1 == (len(nn_weights))

                for num in range(0, self.num_layers-1):
                    W = nn_weights[num]
                    b = nn_biases[num]
                    g = nn_gammas[num]
                    self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True))
                    self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
                    self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))
                    print(" - Load NN parameters successfully...")


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
