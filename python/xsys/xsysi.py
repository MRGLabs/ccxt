import asyncio
from bson.objectid import ObjectId
import xsys.neural
from xsys.db import XSYSDB
from xsys.xi import DFX
'''
cd xsys_ccxt/python
../venv/bin/python
from xsys.xsysi import launch
x,db,asyncio,ObjectId = launch()
'''
def launch():
    db = XSYSDB()
    db.start_db_client()
    return DFX(), db, asyncio, ObjectId