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
from sklearn.ensemble import RandomForestClassifier
from mla_load_data import datasets
import random
import time
import sys
import pickle


if len(sys.argv) != 3:
    print ("Please provide an input data filename and output data mla filename")
    sys.exit() 


forest_classifier = RandomForestClassifier(n_estimators=20)

data = datasets(sys.argv[1])

X = data.get_X()
y = data.get_y()

f = open('train_random_forest_times.csv', 'a')

before_time = time.time()
forest_classifier.fit(X,y)
after_time = time.time()

f.write('{0:.3f},{1:.3f},{2:.3f}\n'.format(before_time,after_time,after_time - before_time));
f.close()

pickle.dump(forest_classifier, open(sys.argv[2], 'wb'))


