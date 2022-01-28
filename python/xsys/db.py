import random
from pymongo import MongoClient, DESCENDING
import sys
from xsys.neural import NeuralExample, NeuralExampleSet
from datetime import datetime
import traceback
import numpy as np
from bson.objectid import ObjectId

'''
import asyncio
from bson.objectid import ObjectId
import neural
from db import XSYSDB
db = XSYSDB()
db.start_db_client()
mlp = db.get_market_limit_pairs(product_id='FIL-USD', N_samples_max=100, do_print=True)
open_orders_dict = db.retrieve_open_orders(product_id='BTC-USD')

db.find_most_recent_prediction(product_id='BTC-USD')
db.find_most_recent_notification(kind='xsys_prediction_notification')
db.find_most_recent_notification(kind='xsys_btc_notification')
db.find_most_recent_alpha_stat(stat_id='alpha_max', resolution_grade='low')

db.store_alpha_stat(stat_id='alpha_max', value=15.9504, time=primary_tz.localize(datetime.now()).isoformat())

db.market_limit_pairs.delete_one({'_id': ObjectId('60c4bc8247515ccccae20485')})
db.delete_alpha_stat(ObjectId('6082cd84127aeedd1e782735'))

D = db.get_neural_examples_since(time_start=107.60)
print(f"{max([d['N_samples'] for d in D])} samples,    {max([float(d['MinuteSpan']) for d in D])/60:.1f} hours")
print([d['N_samples'] for d in D])
print([float(f"{d['MinuteSpan']:.1f}") for d in D])
print(len(D))

import torch
state = torch.load(f'trader_state.BTC-USD.pt'); vwap = state['vwaps'][0]; list(map(np.count_nonzero, [vwap.q_price,vwap.q_size,vwap.q_matches_per_second,vwap.q_vwap]))
plt.plot(0.9*np.array(vwap.q_price))
'''
class XSYSDB:
    def __init__(self, manager=None):
        if manager:
            self.mpm = manager
            self.neural_example_Q_in = self.mpm.Queue()
            self.neural_example_Q_out = self.mpm.Queue()

    def start_db_client(self):
        self.client = MongoClient()
        self.neural_example_collection = self.client.xsysdb.neural_examples
        self.alpha_stats = self.client.xsysdb.alpha_stats
        self.notifications = self.client.xsysdb.notifications
        self.open_orders = self.client.xsysdb.open_orders
        self.predictions = self.client.xsysdb.predictions
        self.market_limit_pairs = self.client.xsysdb.market_limit_pairs
        self.current_prices = self.client.xsysdb.current_prices

    def process_neural_example_Q_out(self):
        if not self.neural_example_Q_out.empty():
            example_set = self.neural_example_Q_out.get()
            self.store_neural_example_set(nes=example_set)
            print(f"[db_store_process()]: Successfully Stored Example Set: {example_set}")

    def store_neural_example_set(self, nes:NeuralExampleSet):
        try:
            D = []
            for e in nes.examples:
                D.append(e.dict_rep())
            self.neural_example_collection.insert_many(D)
        except Exception as e:
            print(f"[XSYSDB]: Error encountered while storing Neural Example Set. {e}")
            for d in D:
                for k,v in d.items():
                    print(f"{k}.shape() = {np.shape(v)}")
            print(traceback.format_exc())

    def delete_neural_example(self, object_id):
        self.neural_example_collection.delete_one({'_id': ObjectId(object_id)})

    def store_alpha_stat(self, document):
        try:
            print(f"[{self.__class__.__name__}.store_alpha_stat()]: document = {document}")
            self.alpha_stats.insert_one(document=document)
        except Exception as e:
            print(traceback.format_exc())

    def store_prediction(self, prediction=None):
        try:
            print(f"[{self.__class__.__name__}.store_prediction()]: prediction = {prediction}")
            self.predictions.insert_one(prediction)
        except Exception as e:
            print(traceback.format_exc())

    def store_market_limit_pair(self, market_limit_pair=None):
        try:
            print(f"[{self.__class__.__name__}.store_market_limit_pair()]: market_limit_pair = {market_limit_pair}")
            self.market_limit_pairs.insert_one(market_limit_pair)
        except Exception as e:
            print(traceback.format_exc())

    def store_notification(self, notification_dict=None):
        try:
            #print(f"[{self.__class__.__name__}.store_notification()]: notification = {notification_dict}")
            self.notifications.insert_one(notification_dict)
        except Exception as e:
            print(f"[XSYSDB]: Error encountered while storing notification. {e}")
            print(f"\tnotification_dict = {notification_dict}")
            print(traceback.format_exc())

    def update_market_limit_pair(self, update_key=None, r_sell=None):
        try:
            d = self.market_limit_pairs.find_one({'update_key':update_key})
            if d != None:
                d.update({'r_sell':r_sell})
                d.update({'update_key': r_sell['id']})
                self.market_limit_pairs.replace_one({'update_key':update_key}, d)
            else:
                raise ValueError('MLP not found')
        except Exception as e:
            print(f"[{self.__class__.__name__}.update_market_limit_pair()]: r_sell = {r_sell}")
            print(traceback.format_exc())

    def update_current_price(self, product_id:str=None, price:float=None):
        try:
            q = {'product_id': product_id}
            d = {
                'product_id': product_id,
                'price':price
            }
            self.current_prices.replace_one(q,d,upsert=True)
        except Exception as e:
            print(f"[{self.__class__.__name__}.update_current_price()]: Error encountered in update_current_price():: {e}")
            print(traceback.format_exc())

    def retrieve_current_prices(self, product_ids=['BTC-USD','ETH-USD','YFI-USD','DOT-USD','FIL-USD','ADA-USD']):
        try:
            product_ids = product_ids if type(product_ids) == type([]) else [product_ids]
            P = {}
            for pid in product_ids:
                q = {'product_id': pid}
                d = self.current_prices.find_one(
                    q,
                    sort=[('_id', DESCENDING)]
                )
                if d:
                    P[pid] = d['price']
            return P
        except Exception as e:
            print(traceback.format_exc())
            return {}

    def update_open_orders(self, product_id=None, open_orders_dict=None):
        try:
            q = {'product_id': product_id}
            open_orders_dict.update(q)
            self.open_orders.replace_one(q,
                                         open_orders_dict,
                                         upsert=True)
        except Exception as e:
            print(f"\topen_orders_dict = {open_orders_dict}")
            print(f"[{self.__class__.__name__}.update_open_orders()]: Error encountered in update_open_orders():: {e}")
            print(traceback.format_exc())

    def retrieve_open_orders(self, product_id=None):
        try:
            q = {'product_id': product_id}
            d = self.open_orders.find_one(
                q,
                sort=[('_id', DESCENDING)]
            )
            return d
        except Exception as e:
            print(traceback.format_exc())
            return None

    def find_most_recent_prediction(self, product_id:str):
        try:
            d = self.predictions.find_one(
                {'product_id': product_id},
                sort=[('_id', DESCENDING)]
            )
            return d
        except Exception as e:
            print(f"product_id: {product_id}")
            print(traceback.format_exc())
            return None

    def find_most_recent_notification(self, product_id:str, kind:str):
        try:
            d = self.notifications.find_one(
                {'kind': kind,
                 'product_id':product_id},
                sort=[('_id', DESCENDING)]
            )
            return d
        except Exception as e:
            print(f"kind: {kind}")
            print(traceback.format_exc())
            return None

    def find_most_recent_market_limit_pair(self, product_id:str):
        try:
            d = self.market_limit_pairs.find_one(
                {'product_id':product_id},
                sort=[('_id', DESCENDING)]
            )
            return d
        except Exception as e:
            print(traceback.format_exc())
            return None

    def find_most_recent_alpha_stat(self,
                                    product_id:str='BTC-USD',
                                    stat_id:str='stat_id',
                                    resolution_grade:str=None):
        try:
            q = {'stat_id': stat_id,
                 'product_id': product_id,
                 'resolution_grade': resolution_grade} if resolution_grade else {'stat_id': stat_id,
                                                                                 'product_id':product_id}
            d = self.alpha_stats.find_one(
                q,
                sort=[('_id', DESCENDING)]
            )
            return d
        except Exception as e:
            print(f"query = {q}")
            print(traceback.format_exc())
            return None

    def delete_alpha_stat(self, object_id):
        self.alpha_stats.delete_one({'_id': ObjectId(object_id)})

    def shutdown(self):
        self.client.close()
        print(f"[XSYSDB.shutdown()]: Successfully closed db client.")


    def get_market_limit_pairs(self,
                               product_id: str,
                               N_samples_max: int=10,
                               do_print: bool=False):
        D = self.market_limit_pairs.find(
            filter={'$and': [
                {"product_id": product_id},
            ]},
            sort=[('_id', DESCENDING)],
            limit=N_samples_max
        )
        mlp = {}
        for d in D:
            mlp[d['r_sell']['id']] = d
            del mlp[d['r_sell']['id']]['_id']
            if do_print:
                print(f"\n\t{d['r_sell']['id']}:---------------------------------------\n\t{mlp[d['r_sell']['id']]}")
        return mlp

    # time_start in days since 1-1-2021
    def get_neural_examples_since(self,
                                  product_id: str,
                                  time_start=105.5370,
                                  N_samples_max=100_000):
        D = self.neural_example_collection.find(
            filter={'$and': [
                {"product_id": product_id},
                {"TimeStart": {'$gt': time_start}},
            ]},
            limit=N_samples_max
        )
        return [d for d in D]

    def get_neural_training_samples(self,
                                    product_id: str,
                                    N_backcast_length=1000,
                                    N_forecast_length=100,
                                    min_minute_span=10,
                                    N_samples_max=100_000,
                                    traintest_split=0.9):

        D = self.neural_example_collection.find(
            filter={'$and': [
                {"product_id": product_id},
                {"MinuteSpan": {'$gt': min_minute_span}},
                {"N_samples": {'$gt': N_backcast_length+N_forecast_length+1}},
            ]},
            limit=N_samples_max
        )
        _ = [d for d in D]
        random.shuffle(_)
        xo, y_bc, y_fc = [], [], []
        N_bcfc = N_backcast_length + N_forecast_length
        for s in _:
            offset_R = np.random.randint(1, s['N_samples']-N_bcfc)
            xo.append(np.transpose([
                np.array(s['Time'][-(N_bcfc+offset_R):-(N_forecast_length+offset_R)]) - s['TimeStart'],
                s['Size'][-(N_bcfc+offset_R):-(N_forecast_length+offset_R)],
                s['MPS'][-(N_bcfc+offset_R):-(N_forecast_length+offset_R)],
                s['VWAP'][-(N_bcfc+offset_R):-(N_forecast_length+offset_R)],
            ]))
            y_bc.append(np.transpose([
                s['Price'][-(N_bcfc+offset_R):-(N_forecast_length+offset_R)],
            ]))
            y_fc.append(np.transpose([
                s['Price'][-(N_forecast_length+offset_R):-offset_R],
            ]))

        # Split data into training and testing datasets.
        c = int(np.round((1-traintest_split)*len(_)))
        xo_train, xo_test = xo[c:], xo[:c]
        y_train_bc, y_train_fc, y_test_bc, y_test_fc = y_bc[c:], y_fc[c:], y_bc[:c], y_fc[:c]

        return np.array(xo_train), np.array(xo_test), \
               np.array(y_train_bc), np.array(y_test_bc), \
               np.array(y_train_fc), np.array(y_test_fc)