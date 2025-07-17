import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from npeet import entropy_estimators as ee
from typing import List, Optional, Literal
from pandas import DataFrame

from .base import BaseFeatureSelector

class CMIFeatureSelector(BaseFeatureSelector):
    """
    Greedy feature selection using Conditional Mutual Information (CMI).
    """
    def __init__(
        self,
    ):
        print("CMI feature selection initialized.")