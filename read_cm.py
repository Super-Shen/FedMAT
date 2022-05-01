import numpy as np
import pandas as pd
cm_pth = "/home/haotian/myCode/FedMAT/FedMAT_code/iLog/w_30_whatdoing_new/user_cms.npy"

cms = np.load(cm_pth, allow_pickle=True)

cms = cms.tolist()
for k in cms.keys():
    print("\n", k)
    print(cms[k])