  # Copyright 2017 Kandice McLean
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd

class datasets:

    X = None
    y = None
    class_names = None
    k_fold =None
    num_features = None
    num_samples = None
    num_classes = None
    feature_names = None

    def __init__(self,filename):
        """Initialise"""

        num_data = pd.read_csv(filename, nrows=1, header=None)
        self.num_features = num_data.iloc[0,1]
        self.num_samples = num_data.iloc[0,0]
        self.num_classes = num_data.iloc[0,2]

        self.feature_names = pd.read_csv(filename, nrows = 1, skiprows=1, header=None)                

        data_df = pd.read_csv(filename, skiprows=2)

        self.X = data_df.iloc[:, 5:-2].values
        self.y = data_df.iloc[:,-1].values
        self.class_names = data_df.iloc[:,-2].values
       
      
    def get_X(self):
        return self.X

    def get_y(self):
        return self.y

    def get_kFold(self):
        return self.k_fold  

    def get_num_features(self):
        return self.num_features

    def get_num_samples(self):
        return self.num_samples

    def get_num_classes(self):
        return self.num_classes

    def get_class_names(self):
        return self.class_names

    def get_feature_names(self):
        return self.class_names

