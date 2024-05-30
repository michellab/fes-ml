import glob

import numpy as np

U_kln = []
frames_disc = 10
step = 1
for i in range(100):
    out = f"OUTPUT_{i}"
    try:
        f = glob.glob(out + "/*npy")[0]
    except:
        break
    U_kln.append(np.load(f)[:, frames_disc::step])

U_kln_array = np.asarray(U_kln)
print(U_kln_array.shape)
np.save("U_kln_agg.npy", U_kln_array)
