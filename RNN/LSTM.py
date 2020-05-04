from __future__ import print_function, division
from builtins import range
import numpy as np

def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    #i coded it during taking cs231n class
    #so the parameters follows as they provide

    next_h, next_c, cache = None, None, None
    N, D = x.shape
    N, H = prev_h.shape
    
    ans = np.dot(x, Wx) + np.dot(prev_h, Wh) + b

    #forgot gate
    f_t = sigmoid(ans[:, 0 * H : 1 * H])
    #update gate
    i_t = sigmoid(ans[:, 1 * H : 2 * H])
    #cell input gate
    c_t_bar = sigmoid(ans[:, 2 * H : 3 * H])
    #cell update gate
    c_t = (f_t * prev_c) + (i_t * c_t_bar)
    #output gate
    o_t = sigmoid(ans[:, 3 * H : 4 * H])
    h_t = o_t * np.tanh(c_t)

    next_h = h_t
    next_c = c_t
    cache = (x, prev_h, prev_c, f_t, i_t, c_t_bar, o_t, Wx, Wh, b, N, D, H)
    
    return next_h, next_c, cache

def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    x, prev_h, prev_c, f_t, i_t, c_t_bar, o_t, Wx, Wh, b, N, D, H = cache
    db= np.zeros((4*H))
    dWh = np.zeros((H, 4*H))
    dWx = np.zeros((D, 4*H))
    dx = np.zeros_like(x)
    #db = np.zeros_like(b)
    
    #stage 1
    dho = np.tanh(dnext_c) * dnext_h
    dhc = o_t * dnext_h
    
    #stage 2: backprop through gates
    #output gate
    d_o = dho * (1 - o_t) * o_t
    #dnext cell
    dc = dhc * (1 - dnext_c **2)
    #forget gate
    df = prev_c * dc * (1 - f_t) * f_t
    #dprev_c
    dprev_c = f_t * dc
    #input gate
    di = c_t_bar * dc * (1 - i_t) * i_t
    #cell gate
    dc_bar = i_t * dc * (1 - c_t_bar) * c_t_bar

    #stage 3: backprop through the weights and biases
    #forget gate
    db[0 * H : 1 * H] = np.sum(df, axis=0)
    dWh[:, 0 * H : 1 * H] = prev_h.T.dot(df)
    dWx[:, 0 * H : 1 * H] = x.T.dot(df)
    dx += df.dot(Wx[:, 0*H:1*H].T)
    dprev_h = df.dot(Wh[:, 0*H:1*H].T)
    #input gate
    db[1 * H : 2 * H] = np.sum(di, axis=0)
    dWh[:, 1 * H : 2 * H] = prev_h.T.dot(di)
    dWx[:, 1 * H : 2 * H] = x.T.dot(di)
    dx += di.dot(Wx[:, 1*H:2*H].T)
    dprev_h += di.dot(Wh[:, 1*H:2*H].T)
    #cell gate
    db[2 * H : 3 * H] = np.sum(dc_bar, axis=0)
    dWh[:, 2 * H : 3 * H] = prev_h.T.dot(dc_bar)
    dWx[:, 2 * H : 3 * H] = x.T.dot(dc_bar)
    dprev_h += dc_bar.dot(Wh[:, 2 * H : 3 * H].T)
    dx += dc_bar.dot(Wx[:, 2 * H : 3 * H].T)
    #out gate
    db[3 * H : 4 * H] = np.sum(d_o, axis=0)
    dWh[:, 3 * H : 4 * H] = prev_h.T.dot(d_o)
    dWx[:, 3 * H : 4 * H] = x.T.dot(d_o)
    dprev_h += d_o.dot(Wh[:, 2 * H : 3 * H].T)
    dx += d_o.dot(Wx[:, 2 * H : 3 * H].T)

    return dx, dprev_h, dprev_c, dWx, dWh, db
