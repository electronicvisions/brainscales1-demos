import sys
import numpy as np
import os
import pyNN.random as rn
import pickle
from neo.io import NeoMatlabIO
from neo import SpikeTrain, Segment, Block
import h5py
from quantities import ms

from datetime import datetime
import os
import numpy as np

import time

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-2s [%(name)s] %(message)s',
                    filename=None,
                    filemode='w')
logger = logging.getLogger("cortical_microcircuit")


# choose simulation backend
try:
    simulator = sys.argv[1]
except IndexError:
    # Default to brainscales
    simulator = "brainscales"

if simulator == "brainscales":
    from pyhalco_hicann_v2 import Wafer, FGBlockOnHICANN
    from pyhalco_common import Enum, iter_all
    import pyhmf as pynn
    from pysthal import Settings
    import pysthal
    from pyhalbe import HICANN

    from pymarocco.results import Marocco
    from pymarocco import Defects
    from pymarocco import PyMarocco
    from pymarocco_runtime import Runtime

    from pysthal.command_line_util import init_logger
    init_logger("ERROR", [])

elif simulator == "nest":
    import pyNN.nest as pynn

else:
    logger.error(f"unknown simulator: {simulator}")
    raise RuntimeError("Unknown simulator")


