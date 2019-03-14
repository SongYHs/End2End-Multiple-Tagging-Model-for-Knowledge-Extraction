#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:52:41 2019

@author: Song Yunhua
"""
from keras.layers.recurrent import GRU,LSTM
from keras.layers.core import Layer
import keras.backend as K
import theano

class LSTM_Decoder(LSTM):
    input_ndim = 3
    def __init__(self, output_length, hidden_dim=None,state_input=True, **kwargs):
        self.output_length = output_length
        self.hidden_dim = hidden_dim
#        self.output_dim=output_dim
        self.state_outputs = []
        self.state_input = state_input
        self.return_sequences = True #Decoder always returns a sequence.
        self.updates = []
        super(LSTM_Decoder, self).__init__(**kwargs)
    def build(self):
        input_shape = self.input_shape
        dim = input_shape[-1]
        self.input_dim = dim
        self.input = K.placeholder(input_shape)
        if not self.hidden_dim:
            self.hidden_dim = dim
        hdim = self.hidden_dim
        if not self.output_dim:
            self.output_dim = dim
        outdim = self.output_dim
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            self.reset_states()
        else:
            self.states = [None, None]
        self.W_i = self.init((outdim, hdim))
        self.U_i = self.inner_init((hdim, hdim))
        self.b_i = K.zeros((hdim))
        self.W_f = self.init((outdim, hdim))
        self.U_f = self.inner_init((hdim, hdim))
        self.b_f = self.forget_bias_init((hdim))
        self.W_c = self.init((outdim, hdim))
        self.U_c = self.inner_init((hdim, hdim))
        self.b_c = K.zeros((hdim))
        self.W_o = self.init((outdim, hdim))
        self.U_o = self.inner_init((hdim, hdim))
        self.b_o = K.zeros((hdim))
        self.W_x = self.init((hdim, outdim))
        self.b_x = K.zeros((outdim))
        self.V_i = self.init((dim, hdim))
        self.V_f = self.init((dim, hdim))
        self.V_c = self.init((dim, hdim))
        self.V_o = self.init((dim, hdim))
        self.trainable_weights = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
            self.W_x,           self.b_x,
            self.V_i, self.V_c, self.V_f, self.V_o
        ]
        self.input_length = self.input_shape[-2]
        if not self.input_length:
            raise Exception ('AttentionDecoder requires input_length.')
    def set_previous(self, layer, connection_map={}):
        self.previous = layer
        self.build()
    def get_initial_states(self, X):
        # build an all-zero tensor of shape (samples, hidden_dim)
        initial_state = K.zeros_like(X)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.hidden_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, hidden_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states
    def ssstep(self,
               h,
              x_tm1,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c, w_i, w_f, w_c, w_o, w_x,  v_i, v_f, v_c, v_o, b_i, b_f, b_c, b_o, b_x):
        xi_t = K.dot(x_tm1, w_i)+ b_i+ K.dot(h, v_i)
        xf_t = K.dot(x_tm1, w_f)  + b_f+ K.dot(h, v_f)
        xc_t = K.dot(x_tm1, w_c) + b_c+ K.dot(h, v_c)
        xo_t = K.dot(x_tm1, w_o)  + b_o+ K.dot(h, v_o)
        i_t = self.inner_activation(xi_t + K.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + K.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + K.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + K.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        x_t =self.activation(K.dot(h_t, w_x) + b_x)
        return x_t, h_t, c_t
    def get_output(self, train=False):
        H = self.get_input(train)
        Hh = K.permute_dimensions(H, (1, 0, 2))
        def rstep(o,index,Hh):
            return Hh[index],index-1
        [RHh,index],update = theano.scan(
        rstep,
        n_steps=Hh.shape[0],
        non_sequences=[Hh],
        outputs_info= [Hh[-1]]+[-1])
        X = K.permute_dimensions(H, (1, 0, 2))[-1]
        outdim=self.output_dim
        X1=X[:,:outdim]+X[:,outdim:]
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)
        [outputs,hidden_states, cell_states], updates = theano.scan(
            self.ssstep,
            sequences=RHh,
            n_steps = self.output_length,
            outputs_info=[X1] + initial_states,
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c,
                          self.W_i, self.W_f, self.W_c, self.W_o,
                          self.W_x, self.V_i, self.V_f, self.V_c,
                          self.V_o, self.b_i, self.b_f, self.b_c,
                          self.b_o, self.b_x])
        states = [hidden_states[-1], cell_states[-1]]
        if self.stateful and not self.state_input:
            self.updates = []
            for i in range(2):
                self.updates.append((self.states[i], states[i]))
        for o in self.state_outputs:
            o.updates = []
            for i in range(2):
                o.updates.append((o.states[i], states[i]))
        return K.permute_dimensions(outputs, (1, 0, 2))
    @property
    def output_shape(self):
        shape = list(super(LSTM_Decoder, self).output_shape)
        shape[1] = self.output_length
        return tuple(shape)
    def get_config(self):
        config = {'name': self.__class__.__name__}
        base_config = super(LSTM_Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class GRU_Decoder(GRU):
    input_ndim = 3
    def __init__(self, output_length, hidden_dim=None,state_input=True, **kwargs):
        self.output_length = output_length
        self.hidden_dim = hidden_dim
        self.state_outputs = []
        self.state_input = state_input
        self.return_sequences = True #Decoder always returns a sequence.
        self.updates = []
        super(GRU_Decoder, self).__init__(**kwargs)
    def build(self):
        input_shape = self.input_shape
        dim = input_shape[-1]
        self.input_dim = dim
        self.input = K.placeholder(input_shape)
        if not self.hidden_dim:
            self.hidden_dim = dim
        hdim = self.hidden_dim
        if not self.output_dim:
            self.output_dim = dim
        outdim = self.output_dim
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            self.reset_states()
        else:
            self.states = [None]
        self.W_r = self.init((outdim, hdim))
        self.U_r = self.inner_init((hdim, hdim))
        self.b_r = K.zeros((hdim))
        self.W_z = self.init((outdim, hdim))
        self.U_z = self.inner_init((hdim, hdim))
        self.b_z = K.zeros((hdim))#self.forget_bias_init((hdim))
        self.W_s = self.init((outdim, hdim))
        self.U_s = self.inner_init((hdim, hdim))
        self.b_s = K.zeros((hdim))
        self.W_x = self.init((hdim, outdim))
        self.b_x = K.zeros((outdim))
        self.V_r = self.init((dim, hdim))
        self.V_z = self.init((dim, hdim))
        self.V_s = self.init((dim, hdim))
        self.trainable_weights = [
            self.W_r, self.U_r, self.b_r,
            self.W_z, self.U_z, self.b_z,
            self.W_s, self.U_s, self.b_s,
            self.W_x,           self.b_x,
            self.V_r, self.V_z, self.V_s
        ]
        self.input_length = self.input_shape[-2]
        if not self.input_length:
            raise Exception ('AttentionDecoder requires input_length.')
    def set_previous(self, layer, connection_map={}):
        self.previous = layer
        self.build()
    def get_initial_states(self, X):
        initial_state = K.zeros_like(X)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.hidden_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, hidden_dim)
        initial_states = [initial_state]# for _ in range(len(self.states))]
        return initial_states
    def ssstep(self,
               h,
              x_tm1,
              s_tm1,
              u_r, u_z, u_s, w_r, w_z, w_s, w_x,  v_r, v_z, v_s, b_r, b_z, b_s, b_x):
        xr_t = K.dot(x_tm1, w_r)+ b_r+ K.dot(h, v_r)
        xz_t = K.dot(x_tm1, w_z)  + b_z+ K.dot(h, v_z)
        r_t  = self.inner_activation(xr_t + K.dot(s_tm1, u_r))
        z_t  = self.inner_activation(xz_t + K.dot(s_tm1, u_z))
        xs_t = K.dot(x_tm1, w_s)  + b_s+ K.dot(h, v_s)
        s1_t = self.activation(xs_t + K.dot(r_t*s_tm1, u_s))
        s_t = (1-z_t) * s_tm1 + z_t * s1_t
        x_t = self.activation(K.dot(s_t, w_x) + b_x)
        return x_t, s_t
    def get_output(self, train=False):
        H = self.get_input(train)
        Hh = K.permute_dimensions(H, (1, 0, 2))
        def rstep(o,index,Hh):
            return Hh[index],index-1
        [RHh,index],update = theano.scan(
        rstep,
        n_steps=Hh.shape[0],
        non_sequences=[Hh],
        outputs_info= [Hh[-1]]+[-1])
        #RHh=K.permute_dimensions(RHh, (1, 0, 2))
        X = K.permute_dimensions(H, (1, 0, 2))[-1]
        #X1=(X[:,:self.input_shape//2,:]+X[:,:self.input_shape//2,:])/2
        outdim=self.output_dim
        X1=X[:,:outdim]+X[:,outdim:]
        
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)
        [outputs,hidden_states], updates = theano.scan(
            self.ssstep,
            sequences=RHh,
            n_steps = self.output_length,
            outputs_info=[X1] + initial_states,#initial_states输入状态改为一个
            non_sequences=[self.U_r, self.U_z, self.U_s,
                          self.W_r, self.W_z, self.W_s, self.W_x,
                          self.V_r, self.V_z, self.V_s,
                          self.b_r, self.b_z, self.b_s, self.b_x])
        states = [hidden_states[-1]]
        if self.stateful and not self.state_input:
            self.updates = []
            for i in range(2):
                self.updates.append((self.states[i], states[i]))
        for o in self.state_outputs:
            o.updates = []
            for i in range(2):
                o.updates.append((o.states[i], states[i]))
        return K.permute_dimensions(outputs, (1, 0, 2))
    @property
    def output_shape(self):
        shape = list(super(GRU_Decoder, self).output_shape)
        shape[1] = self.output_length
        return tuple(shape)
    def get_config(self):
        config = {'name': self.__class__.__name__}
        base_config = super(GRU_Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    
class ReverseLayer2(Layer):
    def __init__(self,layer):
        self.layer=layer
        self.trainable_weights = []
        self.regularizers = []
        self.constraints = []
        self.updates = []
        params, regs, consts, updates =self.layer.get_params()
        self.regularizers += regs
        self.updates += updates
        # params and constraints have the same size
        for p, c in zip(params, consts):
            if p not in self.trainable_weights:
                self.trainable_weights.append(p)
                self.constraints.append(c)
        super(ReverseLayer2, self).__init__()
    @property
    def output_shape(self,train=False):
        return self.layer.output_shape
    def get_output(self, train=False):
         b=self.layer.get_output(train)
         a=b.dimshuffle((1, 0, 2))
         def rstep(o,index,H):
            return H[index],index-1
         [results,index],update = theano.scan(
         rstep,
        n_steps=a.shape[0],
        non_sequences=[a],
        outputs_info= [a[-1]]+[-1])
         results2=results.dimshuffle((1, 0, 2))
         return results2
    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'layers': self.layer.get_config()}
        base_config = super(ReverseLayer2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def get_input(self, train=False):
        res = []
        o = self.layer.get_input(train)
        if not type(o) == list:
            o = [o]
        for output in o:
            if output not in res:
                res.append(output)
        return res
    @property
    def input(self):
        return self.get_input()
    def supports_masked_input(self):
        return False
    def get_output_mask(self, train=None):
        return None