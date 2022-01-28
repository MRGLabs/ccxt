import sys
import pytz
import ccxt as ccxt
from getpass import getpass
from xsys.feist import srtf, feiphi

class DFX:

    def __init__(self,
                 DEBUG=False):
        self.DEBUG = DEBUG
        self.primary_tz = pytz.timezone('US/Eastern')
        self.x = None
        self.prepare_for_trading()

    def prep_1(self):
        self.bp_ = getpass

    def prep_2(self):
        self.x = ccxt.digifinex({'apiKey': self._3,'secret': self._4})

    def prepare_for_trading(self):
        self.prep_1()
        self.prepare_dfx()
        self.prep_2()
        #self.refresh_account()
        print(f"[initialization complete]")

    def prepare_dfx(self):
        _1, _2 = feiphi('xsys/sandbox/tf0.xs', 'xsys/sandbox/tf4.xs', '', '-d')
        self.bp = self.bp_(stream=sys.stderr)
        self._3, self._4 = "".join(srtf(_1,_2)).split('\n')