import tensorflow as tf
import numpy as np
import scipy.io
import time

from utilities import HFM, relative_error


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution() # turning off eager execution

    batch_size = 10000

    layers = [4] + 10*[5*50] + [4]

    # # Load Shear Data
    # data_shear = scipy.io.loadmat('../Data/real_Aorta_shear_nondim.mat')
    
    # xb_star = data_shear['xb_star'] 
    # yb_star = data_shear['yb_star']
    # zb_star = data_shear['zb_star']
    # nx_star = data_shear['nx_star']
    # ny_star = data_shear['ny_star']
    # nz_star = data_shear['nz_star']
    # Sx_star = data_shear['Sx_star']
    # Sy_star = data_shear['Sy_star']
    # Sz_star = data_shear['Sz_star']

    # Load Data
    data = scipy.io.loadmat('../Data/Aorta_nondim.mat')

    t_star = data['t_star'] # T x 1
    x_star = data['x_star'] # N x 1
    y_star = data['y_star'] # N x 1
    z_star = data['z_star'] # N x 1

    T = t_star.shape[0]
    N = x_star.shape[0]

    U_star = data['U_star'] # N x T
    V_star = data['V_star'] # N x T
    W_star = data['W_star'] # N x T
    P_star = data['P_star'] # N x T

    # Rearrange Data
    T_star = np.tile(t_star, (1,N)).T # N x T
    X_star = np.tile(x_star, (1,T)) # N x T
    Y_star = np.tile(y_star, (1,T)) # N x T
    Z_star = np.tile(z_star, (1,T)) # N x T

    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################

    T_data = T
    N_data = N
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_data, replace=False)
    t_data = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_data = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_data = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    z_data = Z_star[:, idx_t][idx_x,:].flatten()[:,None]
    u_data = U_star[:, idx_t][idx_x,:].flatten()[:,None]
    v_data = V_star[:, idx_t][idx_x,:].flatten()[:,None]
    w_data = W_star[:, idx_t][idx_x,:].flatten()[:,None]

    T_eqns = T
    N_eqns = N
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_eqns, replace=False)
    t_eqns = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_eqns = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_eqns = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    z_eqns = Z_star[:, idx_t][idx_x,:].flatten()[:,None]

    # Training
    model = HFM(t_data, x_data, y_data, z_data,
                t_eqns, x_eqns, y_eqns, z_eqns,
                u_data, v_data, w_data,
                layers, batch_size, Rey = 1015, ExistModel=1, uvDir='NS_NN_expand_05_05_2021.pickle')

    ################# Save Data ###########################
    U_pred = 0*U_star
    V_pred = 0*V_star
    W_pred = 0*W_star
    P_pred = 0*P_star
    # Sx_pred = 0*Sx_star
    # Sy_pred = 0*Sy_star
    # Sz_pred = 0*Sz_star
    for snap in range(0,t_star.shape[0]):
        t_test = T_star[:,snap:snap+1]
        x_test = X_star[:,snap:snap+1]
        y_test = Y_star[:,snap:snap+1]
        z_test = Z_star[:,snap:snap+1]

        u_test = U_star[:,snap:snap+1]
        v_test = V_star[:,snap:snap+1]
        w_test = W_star[:,snap:snap+1]
        p_test = P_star[:,snap:snap+1]

        ## Prediction

        # Velocity
        u_pred, v_pred, w_pred, p_pred = model.predict(t_test, x_test, y_test, z_test)
        
        # Shear
        # sx_pred, sy_pred, sz_pred = model.predict_shear(t_test[0] + 0.0*xb_star,
        #                                                 xb_star, yb_star, zb_star,
        #                                                 nx_star, ny_star, nz_star)

        U_pred[:,snap:snap+1] = u_pred
        V_pred[:,snap:snap+1] = v_pred
        W_pred[:,snap:snap+1] = w_pred
        P_pred[:,snap:snap+1] = p_pred

        # Sx_pred[:,snap:snap+1] = sx_pred
        # Sy_pred[:,snap:snap+1] = sy_pred
        # Sz_pred[:,snap:snap+1] = sz_pred

        # Error
        error_u = relative_error(u_pred, u_test)
        error_v = relative_error(v_pred, v_test)
        error_w = relative_error(w_pred, w_test)
        error_p = relative_error(p_pred - np.mean(p_pred), p_test - np.mean(p_test))

        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error w: %e' % (error_w))
        print('Error p: %e' % (error_p))

    # scipy.io.savemat('../Results/Aorta3D_results_rough_weighted_%s.mat' %(time.strftime('%d_%m_%Y')),
    #                  {'U_pred':U_pred, 'V_pred':V_pred, 'W_pred':W_pred, 'P_pred':P_pred,
    #                   'Sx_pred':Sx_pred, 'Sy_pred':Sy_pred, 'Sz_pred':Sz_pred})

    scipy.io.savemat('../Results/Aorta3D_results_normal_coarse_transfer_weighted_%s.mat' %(time.strftime('%d_%m_%Y')),
                     {'U_pred':U_pred, 'V_pred':V_pred, 'W_pred':W_pred, 'P_pred':P_pred})
