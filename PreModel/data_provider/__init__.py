

import pandas as pd
import numpy as np

# *****忽略warning*****
import warnings
warnings.filterwarnings('ignore')

# *****生成文件夹*****
import os
def mkdir(path: str):
	folder = os.path.exists(path)
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径


