# basic imports
import os, random, json
from pathlib import Path
import datetime as dt
import pandas as pd
import numpy as np
import scipy.stats as stats

#SHAP & VIF
import shap
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor

# plotting & visualization
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# sklearn imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score, auc, roc_curve, ConfusionMatrixDisplay,
confusion_matrix, classification_report)
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

#Quant Finance
from quantmod.indicators import *
import pyfolio as pf
from pyfolio import timeseries
from ta import add_all_ta_features

# tensorflow
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Accuracy, AUC, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dropout, Dense, Flatten
from tensorflow.keras.layers import LSTM

# kerastuner
import keras_tuner as kt
from keras_tuner import HyperParameters
from keras_tuner.tuners import BayesianOptimization