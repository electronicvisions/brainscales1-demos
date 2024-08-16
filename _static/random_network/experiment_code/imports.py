# Splitting the files and merging them for execution is a workaround for a bug in the nmpi queue runner
# that does not allow for uploading more than one source file.
# (cf. https://github.com/HumanBrainProject/hbp-neuromorphic-client/issues/21)
from pyNN.random import NumpyRNG
from pyNN.random import RandomDistribution as RD
import pickle
import sys
import logging
from quantities import ms, Hz
import numpy as np
import os
from datetime import datetime
import h5py
from elephant import spike_train_generation
from collections import defaultdict
from neo import Block, SpikeTrain, Segment

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-2s [%(name)s] %(message)s',
                    filename=None,
                    filemode='w')
logger = logging.getLogger("model_random")

try:
    simulator = sys.argv[1]
except IndexError:
    # Default to brainscales
    simulator = "brainscales"

if simulator == "brainscales":
    from pysthal.command_line_util import init_logger
    import pysthal
    from pymarocco_runtime import Runtime
    from pymarocco import PyMarocco
    # Always required even if not used directly.
    # Patches some methodes
    import pymarocco.results
    import pyhmf as pynn
    from pyhalco_hicann_v2 import Wafer, FGBlockOnHICANN
    from pyhalco_common import iter_all, Enum
    from pyhalbe import HICANN
    from pymarocco import Defects
    init_logger("ERROR", [])

elif simulator == "nest":
    import pyNN.nest as pynn

else:
    logger.error(f"unknown simulator: {simulator}")
    raise RuntimeError("Unknown simulator")


