"""
DEM Super Resolution Project - Source Package

基于Depth Anything V2改进的DEM超分辨率模型
"""

__version__ = "1.0.0"

from .scripts.train import train_sr_model
from .scripts.inference import run_dem_super_resolution