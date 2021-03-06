from scipy.signal import detrend
import numpy as np
import xray

def get_data(data_file):

    data = xray.open_dataset(data_file, decode_times=False)

    n_times, n_lat, n_lon = data[data.data_vars.keys()[0]].shape
    p = n_lat * n_lon
    n = n_times
    X = np.zeros((n, p))

    X = np.array(data[data.data_vars.keys()[0]]).reshape(n, -1)

    # remove seasonality
    period = 12
    for m in range(period):

        X[m::period] -= np.mean(X[m::period], axis=0)[None, :]
        X[m::period] /= np.std(X[m::period], axis=0)[None, :]
        if np.sum(np.isnan(X[m::period])) > 0:
            X[m::period] = np.where(np.isnan(X[m::period]), 0, X[m::period])

    # remove trend
    X = detrend(X, axis=0, type='linear')

    return X


def target_region(lx, Lx):

    air_file = 'air.mon.mean.nc'
    pres_file = 'pres.mon.mean.nc'
    pr_wtr_file = 'pr_wtr.mon.mean.nc'
    rhum_file = 'rhum.mon.mean.nc'
    slp_file = 'slp.mon.mean.nc'
    uwnd_file = 'uwnd.mon.mean.nc'
    vwnd_file = 'vwnd.mon.mean.nc'

    air = get_data(air_file)
    pres = get_data(pres_file)
    pr_wtr = get_data(pr_wtr_file)
    rhum = get_data(rhum_file)
    slp = get_data(slp_file)
    uwnd = get_data(uwnd_file)
    vwnd = get_data(vwnd_file)

    n, p = air.shape
    X = np.zeros((n, 7 * (p - 1)))

    pos_lx = int((90 - lx) / 2.5)
    pos_Lx = (np.ceil(Lx / 2.5)).astype(int)
    target = pos_lx * 144 + pos_Lx

    begin = 0
    for j in range(p - 1):

        if j == target:
            continue
        X[:, begin:begin + 7] = np.vstack((air[:, j], pres[:, j],
                                           pr_wtr[:, j], rhum[:, j],
                                           slp[:, j], uwnd[:, j],
                                           vwnd[:, j])).T
        begin += 7

    y = air[:, target]
    # y = pr_wtr[:, target]
    # y /= linalg.norm(y)

    np.save("_Xclimate_design", X)
    np.save("_yclimate_target", y)

    return X, y


lx = 14; Lx = 17 #Dakar
# lx = 48; Lx = 2 #Paris
X, y = target_region(lx, Lx)

