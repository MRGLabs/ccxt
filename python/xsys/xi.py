import sys
import pytz
import ccxt as ccxt
from getpass import getpass
from xsys.feist import srtf, feiphi
import asyncio
import traceback

class DFX:

    def __init__(self,
                 DEBUG=False):
        self.DEBUG = DEBUG
        self.primary_tz = pytz.timezone('US/Eastern')
        self.x = None
        self.prepare_for_trading()

    def refresh_account(self):
        self.balance = self.x.fetchBalance()

    def prep_1(self):
        self.bp_ = getpass

    def prep_2(self):
        self.x = ccxt.digifinex({'apiKey': self._3,'secret': self._4})
        self.markets = self.x.loadMarkets()

    def prepare_for_trading(self):
        self.prep_1()
        self._prepare_dfx()
        self.prep_2()
        #self.refresh_account()
        print(f"[initialization complete]")

    '''
        Example: 
            margin_pnl(leverage=5, order_price=20.5880, amt_filled=11.8198, total_margin=97.35832852, current_price=13.6275, debug=True);
    '''
    def margin_pnl(leverage, hours_in_pos, order_price, amt_filled, initial_margin, current_price, debug=False):
        fee_maker, fee_taker, fee_margin = 0.001, 0.002, 0.0003
        value_initial = order_price * amt_filled
        value_current = current_price * amt_filled
        value_delta = value_current - value_initial
        margin_frozen_initial = value_initial / leverage
        margin_frozen_current = value_current / leverage
        margin_usable = (initial_margin - margin_frozen_initial)
        margin_total = margin_frozen_current + margin_usable
        fee_open = (fee_taker * value_initial)
        fee_close = (fee_taker * value_current)
        fee_interest = (fee_margin * (value_current - margin_total) * divmod(hours_in_pos, 24)[0])
        fee_total = fee_open + fee_close + fee_interest
        pnl = value_delta
        pnl_adjusted = value_delta - (fee_open + fee_close + fee_interest)
        margin_rate = (margin_total + pnl_adjusted) / margin_frozen_current
        margin_closing = initial_margin + pnl_adjusted
        p_liq = order_price * (1 + (.3 / leverage - margin_total / value_initial))
        if debug:
            print(f"MARGIN STATUS:")
            print(f"\tDelta [Spot, Margin]:\t{100 * (current_price - order_price) / order_price:.2f} %,  {100 * pnl_adjusted / initial_margin:.2f} %")
            print(f"\tPnL:\t\t\t{pnl:.2f}\tUSDT")
            print(f"\tPnL-adjusted:\t{pnl_adjusted:.2f}\tUSDT")
            print(f"\tP-LIQ:\t\t\t{p_liq:.2f}\tUSDT")
            print(f"\tMargin Rate:\t{100 * margin_rate:.2f}\t%")
            print(f"\tFrozen Margin:\t{margin_frozen_current:.2f}\tUSDT")
            print(f"\tUsable Margin:\t{margin_usable:.2f}\tUSDT")
            print(f"\tClosing Margin:\t{margin_closing:.2f}\tUSDT")
            print(f"\tFees [Open, Close, Interest, Total]:\t{fee_open:.2f},  {fee_close:.2f},  {fee_interest:.2f},  {fee_total:.2f}\tUSDT")
        return margin_closing, pnl_adjusted, 100 * margin_rate, p_liq

    '''
        cd xsys_ccxt/python
        ../venv/bin/python
        from xsys.xsysi import launch
        x,db,asyncio,ObjectId = launch()
        
        B = x.x.fetchBalance()
        r = x.x.fetch('https://openapi.digifinex.com/v3/trades/symbols')
        r = x.x.fetch('https://openapi.digifinex.com/v3/margin/currencies')
        r = x.x.create_market_sell_order('BTC/USDT',B['BTC']['free'])
        p_fil = x.x.fetchTicker('FIL/BTC')
        r = x.x.create_market_buy_order('FIL/BTC',B['BTC']['free']/p_fil['ask']) # ERROR: this pair does not seem to support market orders

        product_id='BTC/USDT'; funds=1000; predicted_percent_gain=100; hodl_factor=0.333; omlp = asyncio.run(x.place_market_limit_pair(product_id=product_id, funds=funds, predicted_percent_gain=predicted_percent_gain, hodl_factor=hodl_factor, db=db))
    '''
    async def place_market_limit_pair(self,
                                      product_id: str,
                                      funds: float,
                                      predicted_percent_gain: float,
                                      hodl_factor: float = 0.2,
                                      db=None):

        if db == None:
            print(f"[ERROR]: Must pass in an open db to place market limit pair")
            return None

        # hodl_factor multiplies predicted_gain to determine how much profit to hodl in product
        # hodl_factor of 0.5 hodl's half of realized gains
        omlp, f_buy, r_buy, r_sell = None, None, None, None
        try:
            r_buy = self.place_market_order(product_id=product_id, side='buy', funds=f'{funds:.2f}')
            print(f"[{self.__class__.__name__}.place_market_limit_pair()]: 'buy' order executed: {r_buy}")

            async def periodic_fill_check_sell(cbpx, r_buy):
                while True:
                    await asyncio.sleep(0.75)
                    F_raw, F_agg, fee_total = cbpx.get_fills(order_id=r_buy['id'])
                    if r_buy['id'] in F_agg:
                        break
                price = float(F_agg[r_buy['id']]['price']) * (1 + predicted_percent_gain / 1e2)
                size = float(F_agg[r_buy['id']]['size']) * (1 - hodl_factor * predicted_percent_gain / 1e2)
                price = f'{price:.2f}'
                size = f'{size:0.3f}' if product_id == 'FIL-USD' else f'{size:0.6f}' if product_id == 'YFI-USD' else f'{size:0.8f}'
                r_sell = cbpx.place_limit_order(product_id=product_id,
                                                side='sell',
                                                price=price,
                                                size=size)
                print(f"[{self.__class__.__name__}.place_market_limit_pair()]: 'sell' order executed: {r_sell}")
                return F_agg[r_buy['id']], r_sell

            f_buy, r_sell = await periodic_fill_check_sell(self, r_buy)

            if not (f_buy is None):
                omlp = {
                    'product_id': product_id,
                    'update_key': r_sell['id'],
                    'f_buy': f_buy,
                    'r_sell': r_sell
                }
                db.store_market_limit_pair(market_limit_pair=omlp.copy())

            return omlp
        except Exception as e:
            print(f"[{self.__class__.__name__}.place_limit_order_pair()]: {e}")
            print(traceback.format_exc())
        return None

    def _prepare_dfx(self):
        _1, _2 = feiphi('xsys/sandbox/tf0.xs', 'xsys/sandbox/tf4.xs', '', '-d')
        self.bp = self.bp_(stream=sys.stderr)
        self._3, self._4 = "".join(srtf(_1,_2)).split('\n')