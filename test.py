import numpy as np

import dataset_boston
import lineareg


X, y = dataset_boston.load_boston()
lineareg.train(X, y, 1000, 1e-8)

