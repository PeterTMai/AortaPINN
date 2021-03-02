"""
@author: Thanh Trung Mai
"""

import tensorflow as tf
import numpy as np
import scipy.io
import time
import sys

from utilities import neural_net, Navier_Stokes_3D, \
                      tf_session, mean_squared_error, relative_error


class HFM(object):
    # notational conventions
    # _tf: placeholders for input/output data and points used to regress the equations
    # _pred: output of neural network
    # _eqns: points used to regress the equations
    # _data: input-output data
    # _star: preditions
    
    def __init__(self, t_data, x_data, y_data, z_data,
                       t_eqns, x_eqns, y_eqns, z_eqns,
                       u_data, v_data, w_data,
                       layers, batch_size, Rey):
        
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
        
        # physics "uninformed" neural networks
        self.net_cuvwp = neural_net(self.t_data, self.x_data, self.y_data, self.z_data, layers = self.layers)
        
        [self.u_data_pred,
         self.v_data_pred,
         self.w_data_pred,
         self.p_data_pred] = self.net_cuvwp(self.t_data_tf,
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
        
        # loss
        self.loss = mean_squared_error(self.u_data_pred, self.u_data_tf) + \
                    mean_squared_error(self.v_data_pred, self.v_data_tf) + \
                    mean_squared_error(self.w_data_pred, self.w_data_tf) + \
                    mean_squared_error(self.e1_eqns_pred, 0.0) + \
                    mean_squared_error(self.e2_eqns_pred, 0.0) + \
                    mean_squared_error(self.e3_eqns_pred, 0.0) + \
                    mean_squared_error(self.e4_eqns_pred, 0.0) + \
                    mean_squared_error(self.e5_eqns_pred, 0.0)
        
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