import warnings

import time
from datetime import datetime, timedelta
import numpy as np
import pytz
from nbeats_keras.model import NBeatsNet

warnings.filterwarnings(action='ignore', message='Setting attributes')

class NeuralExample:
    def __init__(self,
                 product_id,
                 vwap_id, seconds_per_bin,
                 T, P, S, VWAP, MPS, MsgPS,
                 T_LR=None, P_LR=[], S_LR=[], VWAP_LR=[], MPS_LR=[], MsgPS_LR=[]):
        self.product_id = product_id
        self.vwap_id = vwap_id
        self.seconds_per_bin = seconds_per_bin
        self.T = T
        self.P = P
        self.S = S
        self.VWAP = VWAP
        self.MPS = MPS
        self.MsgPS = MsgPS if len(P) == len(MsgPS) else np.concatenate([np.mean(MsgPS) * np.ones(len(P) - len(MsgPS)), MsgPS])
        if len(P_LR):
            self.P_LR = P_LR
            self.S_LR = S_LR
            self.VWAP_LR = VWAP_LR
            self.MPS_LR = MPS_LR
            self.MsgPS_LR = MsgPS_LR
        self.prune_empty_bins()

    def prune_empty_bins(self):
        non_zero_ndx = np.argwhere(np.bitwise_and(self.P>0, self.MPS>0))
        self.T = self.T[non_zero_ndx].flatten()
        self.P = self.P[non_zero_ndx].flatten()
        self.S = self.S[non_zero_ndx].flatten()
        self.VWAP = self.VWAP[non_zero_ndx].flatten()
        self.MPS = self.MPS[non_zero_ndx].flatten()
        self.MsgPS = self.MsgPS[non_zero_ndx].flatten()
        if hasattr(self,'P_LR'):
            self.P_LR = self.P_LR[non_zero_ndx].flatten()
            self.S_LR = self.S_LR[non_zero_ndx].flatten()
            self.VWAP_LR = self.VWAP_LR[non_zero_ndx].flatten()
            self.MPS_LR = self.MPS_LR[non_zero_ndx].flatten()
            self.MsgPS_LR = self.MsgPS_LR[non_zero_ndx].flatten()

        # remove what appears to be price data error (single match flash crash)
        ndx_remove = np.argwhere(np.diff(self.P)<-0.20*np.max(self.P))
        if len(ndx_remove):
            ndx_remove += 1
            self.T = np.delete(self.T, ndx_remove)
            self.P = np.delete(self.P, ndx_remove)
            self.S = np.delete(self.S, ndx_remove)
            self.VWAP = np.delete(self.VWAP, ndx_remove)
            self.MPS = np.delete(self.MPS, ndx_remove)
            self.MsgPS = np.delete(self.MsgPS, ndx_remove)
            if hasattr(self,'P_LR'):
                self.P_LR = np.delete(self.P_LR, ndx_remove)
                self.S_LR = np.delete(self.S_LR, ndx_remove)
                self.VWAP_LR = np.delete(self.VWAP_LR, ndx_remove)
                self.MPS_LR = np.delete(self.MPS_LR, ndx_remove)
                self.MsgPS_LR = np.delete(self.MsgPS_LR, ndx_remove)

    # used for storing in db
    # assumed this is run in separate process that is not time critical
    def dict_rep(self):
        if hasattr(self,'P_LR'):
            return {
                "product_id":self.product_id,
                "vwap_id":self.vwap_id,
                "seconds_per_bin":self.seconds_per_bin,
                "N_samples":len(self.T),
                "AlphaMax":np.log2(max(1,np.max(self.MPS))),
                "TimeStart":self.T[0],
                "TimeEnd":self.T[-1],
                "MinuteSpan":len(self.T)*self.seconds_per_bin/60,
                "AvgProductPerMinute": np.nansum(self.S) / max(1e-6, 24 * 60 * (np.max(self.T) - np.min(self.T))),
                "PricePercentSwing": 1e2 * (np.max(self.P) - np.min(self.P)) / (self.P[-1] if self.P[-1] else self.VWAP[-1]),
                "Time":self.T.tolist(),
                "Price":self.P.tolist(),
                "Size":self.S.tolist(),
                "VWAP":self.VWAP.tolist(),
                "MPS":self.MPS.tolist(),
                "MsgPS":self.MsgPS.tolist(),
                "Price_LR":self.P_LR.tolist(),
                "Size_LR":self.S_LR.tolist(),
                "VWAP_LR":self.VWAP_LR.tolist(),
                "MPS_LR":self.MPS_LR.tolist(),
                "MsgPS_LR":self.MsgPS_LR.tolist(),
            }
        else:
            return {
                "product_id":self.product_id,
                "vwap_id":self.vwap_id,
                "seconds_per_bin":self.seconds_per_bin,
                "N_samples":len(self.T),
                "AlphaMax":np.log2(max(1,np.max(self.MPS))),
                "TimeStart":self.T[0],
                "TimeEnd":self.T[-1],
                "MinuteSpan":len(self.T)*self.seconds_per_bin/60,
                "AvgProductPerMinute": np.nansum(self.S) / max(1e-6, 24 * 60 * (np.max(self.T) - np.min(self.T))),
                "PricePercentSwing": 1e2 * (np.max(self.P) - np.min(self.P)) / (self.P[-1] if self.P[-1] else self.VWAP[-1]),
                "Time":self.T.tolist(),
                "Price":self.P.tolist(),
                "Size":self.S.tolist(),
                "VWAP":self.VWAP.tolist(),
                "MPS":self.MPS.tolist(),
                "MsgPS":self.MsgPS.tolist(),
            }

    def __str__(self):
        vwap_delta = self.VWAP[-1]-self.VWAP[0] if len(self.VWAP) else 0
        s = f"[NE]:" \
            f"\n\tN_samples={len(self.T)}" \
            f"\tT[0]={self.T[0]:.4f}, " \
            f"\tMinutes(T)={24*60*(self.T[-1]-self.T[0]):.3f}, " \
            f"\n\tMean(P)={np.mean(self.P):.2f}, " \
            f"\tVWAP[-1]={self.VWAP[-1]:.2f}, " \
            f"\tSum(S)={np.nansum(self.S):.3f}, " \
            f"\tDelta(VWAP)={vwap_delta:.2f}, " \
            f"\tPercentSwing(P)={1e2 * (np.max(self.P) - np.min(self.P)) / (self.P[-1] if self.P[-1] else self.VWAP[-1]):.3f}, "\
            f"\tMean(MSP) = {np.mean(self.MPS):.2f}"
        return s

class NeuralExampleSet:
    def __init__(self,
                 product_id,
                 N_time_steps,
                 vwap_high_res,
                 vwap_low_res=None):

        self.examples = []

        # Multiscale Examples must be of same length
        if vwap_low_res and len(vwap_high_res.q_price) != len(vwap_low_res.q_price):
            return

        self.N_time_steps_per_example = N_time_steps
        self.primary_tz = pytz.timezone('US/Eastern')
        self.T_2021 = self.primary_tz.localize(datetime(2021,1,1))

        ne = NeuralExample(
            product_id=product_id,
            vwap_id=f"{vwap_high_res.name}--{vwap_low_res.name}" if vwap_low_res else vwap_high_res.name,
            seconds_per_bin=vwap_high_res.seconds_per_bin,
            T=np.array([self.days_since_T_2021(t) for t in list(vwap_high_res.q_time)][-N_time_steps:]),

            P=np.array(list(vwap_high_res.q_price)[-N_time_steps:]),
            S=np.array(list(vwap_high_res.q_size)[-N_time_steps:]),
            VWAP=np.array(list(vwap_high_res.q_vwap)[-N_time_steps:]),
            MPS=np.array(list(vwap_high_res.q_matches_per_second)[-N_time_steps:]),
            MsgPS=np.array(list(vwap_high_res.q_msgs_per_second)[-N_time_steps:]),

            P_LR=np.array(list(vwap_low_res.q_price)[-N_time_steps:]) if vwap_low_res else [],
            S_LR=np.array(list(vwap_low_res.q_size)[-N_time_steps:]) if vwap_low_res else [],
            VWAP_LR=np.array(list(vwap_low_res.q_vwap)[-N_time_steps:]) if vwap_low_res else [],
            MPS_LR=np.array(list(vwap_low_res.q_matches_per_second)[-N_time_steps:]) if vwap_low_res else [],
            MsgPS_LR=np.array(list(vwap_low_res.q_msgs_per_second)[-N_time_steps:]) if vwap_low_res else [],
        )
        if len(ne.T):
            self.examples.append(ne)

    def __str__(self):
        s = f""
        for e in self.examples:
            s += f"\n" + e.__str__()
        return s

    def days_since_T_2021(self, dt=None):
        if dt is None:
            dt = datetime.now().astimezone(self.primary_tz)
        return (dt - self.T_2021).total_seconds() / (24 * 3600)

from xsys.db import XSYSDB
class XSYSNeuralModel:
    def __init__(self,
                 product_id:str,
                 xsysdb = None,
                 N_training_samples_max = 100_000,
                 exo_dim = 4,
                 N_backcast_length=900,
                 N_forecast_length=300,
                 hidden_layer_units=128,
                 nb_blocks_per_stack=3,
                 nb_harmonics=10,
                 thetas_dim=(4, 8),
                 share_weights_in_stack=True,
                 traintest_split=0.9):
        self.product_id = product_id
        self.N_training_samples_max = N_training_samples_max
        self.exo_dim = exo_dim
        self.N_backcast_length = N_backcast_length
        self.N_forecast_length = N_forecast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.traintest_split = traintest_split
        self.nb_harmonics = nb_harmonics
        self.thetas_dim = thetas_dim
        self.share_weights_in_stack = share_weights_in_stack

        self.load_model()

        if not xsysdb:
            self.xsysdb = XSYSDB()
            self.xsysdb.start_db_client()
        else:
            self.xsysdb = xsysdb

    def load_model(self):
        self.net = NBeatsNet(
            exo_dim=self.exo_dim,
            backcast_length=self.N_backcast_length, forecast_length=self.N_forecast_length,
            stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK),
            nb_blocks_per_stack=self.nb_blocks_per_stack, thetas_dim=self.thetas_dim,
            share_weights_in_stack=self.share_weights_in_stack,
            hidden_layer_units=self.hidden_layer_units, nb_harmonics=self.nb_harmonics
        )

        # Definition of the objective function and the optimizer.
        self.net.compile(loss='mae', optimizer='adam')

        try:
            model = NBeatsNet.load(f'weights/xsys_neural_model.{self.product_id}.'
                                   f'{self.N_backcast_length}.'
                                   f'{self.N_forecast_length}.'
                                   f'{self.hidden_layer_units}.'
                                   f'{self.nb_blocks_per_stack}.'
                                   f'{self.exo_dim}.'
                                   f'{self.nb_harmonics}.'
                                   f'{self.thetas_dim[0]}.'
                                   f'{self.thetas_dim[1]}.'
                                   f'{self.share_weights_in_stack}.'
                                   f'h5')
            self.net = model
        except:
            try:
                model = NBeatsNet.load(f'weights/xsys_neural_model.{self.product_id}.'
                                       f'{self.N_backcast_length}.'
                                       f'{self.N_forecast_length}.'
                                       f'{self.hidden_layer_units}.'
                                       f'{self.nb_blocks_per_stack}.'
                                       f'{self.exo_dim}.'
                                       f'{self.nb_harmonics}.'
                                       f'{self.thetas_dim[0]}.'
                                       f'{self.thetas_dim[1]}.'
                                       f'{self.share_weights_in_stack}.'
                                       f'backup.h5')
                self.net = model
            except:
                print(f"Failed to load saved neural model. Training from scratch")

    def get_training_samples(self,
                             product_id:str,
                             min_minute_span=20):
        return self.xsysdb.get_neural_training_samples(product_id=product_id,
                                                       N_backcast_length=self.N_backcast_length,
                                                       N_forecast_length=self.N_forecast_length,
                                                       min_minute_span=min_minute_span,
                                                       N_samples_max=self.N_training_samples_max,
                                                       traintest_split=self.traintest_split)

    def train_continuously(self,
                           N_epochs_per_loop=20,
                           N_batch_size=128,
                           N_loops_per_save=100):

        n = 0
        while True:
            xo_train, xo_test, \
            y_train_bc, y_test_bc, \
            y_train_fc, y_test_fc = self.get_training_samples(product_id=self.product_id,
                                                              min_minute_span=20)
            print(f"Shape(xo_train, xo_test): {xo_train.shape}, {xo_test.shape}")
            print(f"Shape(y_train_bc, y_test_bc): {y_train_bc.shape}, {y_test_bc.shape}")
            print(f"Shape(y_train_fc, y_test_fc): {y_train_fc.shape}, {y_test_fc.shape}")
            if len(y_train_fc.shape)==3 and len(y_test_fc.shape)==3:
                self.net.fit(x=[y_train_bc, xo_train],
                             y=y_train_fc,
                             validation_data=([y_test_bc, xo_test], y_test_fc),
                             epochs=N_epochs_per_loop,
                             batch_size=N_batch_size,
                             shuffle=True)
                n += 1
                if n % N_loops_per_save == 0:
                    self.net.save(f'weights/xsys_neural_model.{self.product_id}.'
                                  f'{self.N_backcast_length}.'
                                  f'{self.N_forecast_length}.'
                                  f'{self.hidden_layer_units}.'
                                  f'{self.nb_blocks_per_stack}.'
                                  f'{self.exo_dim}.'
                                  f'{self.nb_harmonics}.'
                                  f'{self.thetas_dim[0]}.'
                                  f'{self.thetas_dim[1]}.'
                                  f'{self.share_weights_in_stack}.'
                                  f'h5')
                    self.net.save(f'weights/xsys_neural_model.{self.product_id}.'
                                  f'{self.N_backcast_length}.'
                                  f'{self.N_forecast_length}.'
                                  f'{self.hidden_layer_units}.'
                                  f'{self.nb_blocks_per_stack}.'
                                  f'{self.exo_dim}.'
                                  f'{self.nb_harmonics}.'
                                  f'{self.thetas_dim[0]}.'
                                  f'{self.thetas_dim[1]}.'
                                  f'{self.share_weights_in_stack}.'
                                  f'backup.h5')

if __name__ == '__main__':
    '''
    PYTHONUNBUFFERED=1;CUDA_VISIBLE_DEVICES=0,1;LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64
    '''
    product_id='BTC-USD'
    xnm = XSYSNeuralModel(product_id=product_id,
                          N_backcast_length=900,
                          N_forecast_length=300,
                          hidden_layer_units=128,
                          nb_blocks_per_stack=3)
    xnm.train_continuously(N_epochs_per_loop=100,
                           N_batch_size=200,
                           N_loops_per_save=20)