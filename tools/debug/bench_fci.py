from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
import numpy as np, time
np.random.seed(0)
X = np.random.randn(300, 22)
t0=time.time()
G,_=fci(X, alpha=0.05, independence_test_method=fisherz, verbose=False, show_progress=False)
print('FCI 300 time', time.time()-t0, flush=True)
