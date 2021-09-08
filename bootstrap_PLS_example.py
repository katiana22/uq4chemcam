# LICENSE
# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The U.S. Government has granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly, display publicly, and to permit others to do so.

#Permission is granted to the public to copy, use, modify, and distribute this material without charge, provided that the Notice in its entirety, including without limitation the statement of reserved government rights, are reproduced on all copies.


# %%
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% Settings
n_boot = 100    # number of bootstrap replications
size_boot = 500 # should probably be close to y_train.shape[0]
x_ind = 1       # which oxide weight index

np.random.seed(42)

#%% Load and process data
ox_wt_names = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O']

data_path = '/projects/uq4ml/data/ChemCam/calib_new/'
data = pd.read_csv('%s/merged_comp_data_nona_nodup.csv' % data_path)
wav = pd.read_csv('%s/Supplement_MnO_Cal_Input_outliers_wvl.csv' % data_path, nrows=1)
use_col = [col_name for col_name in list(wav.columns) if col_name.startswith('wvl')]
wav = wav[use_col].values.squeeze()

train_data = data.loc[data['train'] == 'train', :]
test_data = data.loc[data['train'] == 'test', :]

x_train = train_data[ox_wt_names].values
x_test = test_data[ox_wt_names].values
y_col = [cn for cn in list(train_data) if cn.startswith('wvl')]
y_train = train_data[y_col].values
y_test = test_data[y_col].values

uv_ind = np.logical_and(wav >= 246.635, wav <= 338.457)
vio_ind = np.logical_and(wav >= 382.13, wav <= 473.184)
vnir_ind = np.logical_and(wav >= 492.427, wav <= 849.)
good_ind = np.logical_or(np.logical_or(uv_ind, vio_ind), vnir_ind)
wav = wav[good_ind]

y_train[:, uv_ind] = y_train[:, uv_ind] / np.sum(y_train[:, uv_ind], 1, keepdims=True)
y_train[:, vio_ind] = y_train[:, vio_ind] / np.sum(y_train[:, vio_ind], 1, keepdims=True)
y_train[:, vnir_ind] = y_train[:, vnir_ind] / np.sum(y_train[:, vnir_ind], 1, keepdims=True)

y_test[:, uv_ind] = y_test[:, uv_ind] / np.sum(y_test[:, uv_ind], 1, keepdims=True)
y_test[:, vio_ind] = y_test[:, vio_ind] / np.sum(y_test[:, vio_ind], 1, keepdims=True)
y_test[:, vnir_ind] = y_test[:, vnir_ind] / np.sum(y_test[:, vnir_ind], 1, keepdims=True)

#%% Bootstrap PLS for one oxide wt
parameters = {'n_components': np.arange(1, 10)}
boot_preds = []
comps = []
for bi in tqdm(range(n_boot), desc='Bootstrap resampling'):
    # Draw bootstrap sample
    boot_ind = np.random.choice(y_train.shape[0], size=size_boot, replace=True)
    y_boot = y_train[boot_ind, :]
    x_boot = x_train[boot_ind, x_ind]
    # Cross-validation to select number of components
    pls = PLSRegression(scale=False)
    cvres = GridSearchCV(pls, parameters).fit(y_boot, x_boot)
    n_comp = cvres.best_params_['n_components']
    comps.append(n_comp)
    # Fit model
    pls_fit = PLSRegression(scale=False, n_components=n_comp).fit(y_boot, x_boot)
    # Get residuals
    resid = x_boot - pls_fit.predict(y_boot).squeeze()
    # TODO: as a tweak, bin residuals? or estimate heteroskedastic variance trend...?
    # some residuals much higher (for high SiO2, maybe) so could be overly conservative
    # Calculate predictions on test set
    boot_preds.append(pls_fit.predict(y_test).squeeze() + np.random.choice(resid, size=y_test.shape[0]))

# %%
boot_preds = np.array(boot_preds) # shape (n_boot, n_test)
comps = np.array(comps)

# %%
plt.figure(figsize=(6, 4))
plt.hist(comps)
plt.xlabel('Number of PLS components chosen')

# %%
plt.figure(figsize=(12, 6))
plt.boxplot(boot_preds)
plt.plot(np.arange(1, y_test.shape[0]+1), x_test[:, x_ind], 'g*')
plt.xlabel('Test set index')
plt.ylabel('Distribution of bootstrap predictions')
plt.title('Bootstrap PLS test predictions for %s' % ox_wt_names[x_ind])
# %%
