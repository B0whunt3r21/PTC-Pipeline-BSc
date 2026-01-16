import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from uncertainties import ufloat, nominal_value, unumpy
import numpy as np
import pandas as pd
import os
import argparse
from lmfit  import Model, models, report_fit, Parameter, printfuncs

import DataManager as DM



'''
    @Params:
        dm.. DataManager Module
        path.. relative file path
    @return
        bias_df.. bias dataframe
        dark_df.. dark dataframe
        flat_df.. flat field dataframe
'''
def load_datasets(dm, path):
    """
    Reads all FITS files using your DataManager and returns:
    - bias_df
    - dark_df
    - flat_df
    """
    df = dm.fetchFiles(path)
    df["type"] = df["split_names"].apply(lambda x: x[0])

    bias_df = df[df["type"] == "bias"].copy()

    dark_df = df[df["type"] == "dark"].copy()
    dark_df = dark_df.sort_values(["temp"])

    flat_df = df[df["type"] == "light"].copy()
    flat_df = flat_df.sort_values(["exp"])

    return bias_df, dark_df, flat_df


'''
    @Params:
        frames.. Arbitrary data column
    @return
        stacked.. median stacked frames column
'''
def median_stack(frames):
    return np.median(np.stack(frames), axis=0)


'''
    @Params:
        bias_df.. bias Data
    @return
        master_bias.. stacked master bias
'''
def build_master_bias(bias_df):
    frames = list(bias_df["data"])
    return median_stack(frames)


'''
@Param:
    img1.. the image on which che constant gets added
    img2.. image te be substracted from img1
@return
        diffIMG.. differenciated frame
'''
def constantDifferentiating(img1, img2, const=1000):
    constImg = img1.copy() + const
    diffIMG = constImg - img2.copy()
    return diffIMG




'''
______________________________________________________
|                    PTC Pipeline                    |
|____________________________________________________|
'''

'''
    @Params:
        bias_df.. Bias Data
        center.. [x, y] central point of mask
        dx.. half width of mask
        dy.. half heigth of mask
    @return
        read_noise_dn.. read noise from bias differentiating
'''
def compute_read_noise(bias_df, center, dx, dy):
    b1 = dm.rectangularSelection(bias_df.iloc[0]["data"], center, dx, dy)
    b2 = dm.rectangularSelection(bias_df.iloc[1]["data"], center, dx, dy)
    return np.std(constantDifferentiating(b1, b2)/np.sqrt(2))


'''
    @Params:
        flat_df.. flat data
        master_bias.. stacked master bias
        center.. [x, y] central point of mask
        dx.. half width of mask
        dy.. half heigth of mask
    @return
        ptc_df.. Basic PTC data
'''
def compute_ptc(flat_df, master_bias, center, dx, dy):
    rows = []

    for exp, group in flat_df.groupby("exp"):
        I1 = group["data"].iloc[0] - master_bias
        I2 = group["data"].iloc[-1] - master_bias

        roi1 = dm.rectangularSelection(I1, center, dx, dy)
        roi2 = dm.rectangularSelection(I2, center, dx, dy)

        signal = roi1.mean()
        delta_noise = np.std(roi1 - roi2)/np.sqrt(2)

        rows.append({
            "exp": exp,
            "signal": signal,
            "delta_noise": delta_noise
        })

    return pd.DataFrame(rows).sort_values("exp")


'''
    @Params:
        ptc_df.. PTC Data
        read_noise_dn.. read noise in ADU/DN
    @return
        ptc_df.. PTC Dataframe with shoit noise column
'''
def add_shot_noise(ptc_df, read_noise_dn):
    ptc_df["shot_noise"] = np.sqrt(ptc_df["delta_noise"]**2 - read_noise_dn**2)
    valid = ptc_df["shot_noise"] > 0
    ptc_df = ptc_df[valid]
    return ptc_df


'''
    @Params:
        ptc_df.. PTC Data
    @return
        gain.. calculated gain value
'''
def fit_gain(ptc_df):
    x = ptc_df["signal"].values
    y = ptc_df["shot_noise"].values**2
    slope, intercept = np.polyfit(x, y, 1)
    gain = 1 / slope
    return gain


'''
    @Params:
        flat_df.. Flat data
        master_bias.. stacked master bias
        center.. [x, y] central point of mask
        dx.. half width of mask
        dy.. half heigth of mask
    @return
        prnu.. calculated Pixel response non-unifomrity
'''
def compute_prnu(flat_df, master_bias, center, dx, dy):
    rows = []
    for exp, group in flat_df.groupby("exp"):
        I1 = group["data"].iloc[0] - master_bias
        roi = dm.rectangularSelection(I1, center, dx, dy)
        spatial_std = roi.std()
        signal = roi.mean()
        rows.append((signal, spatial_std))

    df = pd.DataFrame(rows, columns=["signal", "spatial_std"])
    prnu, _ = np.polyfit(df["signal"], df["spatial_std"], 1)
    return prnu


'''
    @Params:
        ptc_df.. PTC Data
        prnu.. Pixel Respinse nun-uniformity from PTC
    @return
        ptc_df.. with added Fixed Pattern Noise
'''
def add_fpn(ptc_df, prnu):
    ptc_df["fpn"] = prnu * ptc_df["signal"]
    return ptc_df


'''
    @Params:
        ptc_df.. PTC Data
        prnu.. Pixel Respinse nun-uniformity from PTC
    @return
        ptc_df.. with added Fixed Pattern Noise
'''
def add_totalNoise(ptc_df):
    ptc_df["total_noise"] = ptc_df["delta_noise"] + ptc_df["fpn"]
    return ptc_df


'''
    @Params:
        ptc_df.. PTC Data
        read_noise.. Read-noise extracted from bias frames
        path.. root path to output folder
'''
def plot_ptc(ptc_df, read_noise, path):
    fitModel = models.LinearModel()

    idx_max = ptc_df["fpn"].idxmax()+1
    xFPN = ptc_df["signal"].iloc[:idx_max]
    yFPN = ptc_df["fpn"].iloc[:idx_max]
    fpn_params = fitModel.guess(yFPN, x=xFPN)
    fpn_result = fitModel.fit(yFPN, fpn_params, x=xFPN, calc_covar=True)

    fpn_x_intercept = - fpn_result.params['intercept'].value / fpn_result.params['slope'].value
    print("X intercept FPN fit: ", fpn_x_intercept)


    idx_max = ptc_df["shot_noise"].idxmax()+1
    xShot = ptc_df["signal"].iloc[:idx_max]
    yShot = ptc_df["shot_noise"].iloc[:idx_max] ** 2
    shot_params = fitModel.guess(yShot, x=xShot)
    shot_result = fitModel.fit(yShot, shot_params, x=xShot, calc_covar=True)

    shot_x_intercept = - shot_result.params['intercept'].value / shot_result.params['slope'].value
    print("X intercept Shot fit: ", 1/shot_x_intercept)


    plt.figure(figsize=(6, 6))
    plt.hlines(read_noise, 0, ptc_df["signal"].min(), "g", label="Read noise")

    plt.loglog(ptc_df["signal"], ptc_df["delta_noise"], "gs", label="Temporal noise")

    plt.loglog(ptc_df["signal"], ptc_df["fpn"], "yo", label="FPN")
    plt.loglog(xFPN, fpn_result.best_fit, 'y--', label='FPN best fit')

    plt.loglog(ptc_df["signal"], ptc_df["shot_noise"], "k.", label="Shot noise")
    plt.loglog(xShot, np.sqrt(shot_result.best_fit), 'k--', label='shot noise best fit')

    plt.loglog(ptc_df["signal"], ptc_df["total_noise"], "rx", label="Total noise")

    plt.xlabel("Signal (DN)")
    plt.ylabel("Noise (DN)")
    plt.legend()
    plt.grid(True)
    plt.minorticks_on()
    plt.title("PTC")

    plt.savefig(f"{path}PTC.svg")


'''
    @Params:
        ptc_df.. PTC Data
        path.. root path to output folder
'''
def plot_nonLinearity(ptc_df, path):
    fitModel = models.LinearModel()

    idx_max = ptc_df["signal"].idxmax()+1
    x = ptc_df["exp"].iloc[:idx_max]
    y = ptc_df["signal"].iloc[:idx_max]
    params = fitModel.guess(y, x=x)
    result = fitModel.fit(y, params, x=x, calc_covar=True)
    nonLin = np.abs(y - result.best_fit)/result.best_fit * 100

    fig, axes = plt.subplots(2, 1, figsize=(6, 5))
    fig.tight_layout()
    fig.suptitle("Nonlinearity")

    axes[0].plot(ptc_df["exp"], ptc_df["signal"], "kx--", label="Signals")
    axes[0].plot(x, result.best_fit, 'b--', label='best fit')
    axes[0].set_xlabel("Exposure (s)")
    axes[0].set_ylabel("Signal (DN)")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].minorticks_on()

    axes[1].plot(x, nonLin, 'k--', label="Nonlinearity")
    axes[1].set_xlabel("Exposure (s)")
    axes[1].set_ylabel("Non-linearity (%)")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].minorticks_on()

    fig.savefig(f"{path}Nonlinearity.svg")



'''
__________________________________________________________________
|                    Dark Current Calculation                    |
|________________________________________________________________|
'''

'''
    @Params:
        dark_df.. Total dark Data
        master_bias.. stacked master bias frame
        gain.. Gain from PTC
        center.. [x, y] central point of mask
        dx.. half width of mask
        dy.. half heigth of mask
    
    @return:
        dc_df.. dataframe containing dark current data
'''
def compute_dark_current(dark_df, master_bias, gain, center, dx, dy):
    rows = []
    
    for temp, group in dark_df.groupby("temp"):
        # Bias-subtracted frames
        
        frames = [dm.rectangularSelection(f - master_bias, center, dx, dy) for f in group["data"]]

        # Median stack if more than one frame
        if len(frames) > 1:
            stacked = median_stack(frames)
        else:
            stacked = frames[0]

        mean_dn = stacked.mean()
        mean_e = mean_dn * gain
        dark_current = mean_e / 60  #60s fixed exposure

        rows.append({
            "temp": temp,
            "mean_dn": mean_dn,
            "mean_e": mean_e,
            "dark_current": dark_current
        })

    return pd.DataFrame(rows).sort_values("temp")


'''
    @Params:
        dc_df.. Dark current dataframe
        path.. root path to output folder
'''
def plot_dark_current(dc_df, path):
    #fitModel = models.ExponentialModel()
    fitModel = models.PolynomialModel()
    params = fitModel.guess(dc_df['dark_current'], x=dc_df["temp"])
    result = fitModel.fit(dc_df['dark_current'], params, x=dc_df["temp"], calc_covar=True)
    print("Dark current at 25: ", result.eval(x=25))

    plt.figure(figsize=(6, 4))
    plt.plot(dc_df["temp"], dc_df["dark_current"], "x", label="Dark current")
    plt.plot(dc_df["temp"], result.best_fit, 'k--', label='best fit')
    plt.xlabel("Temperature $(°C)$")
    plt.ylabel("Dark current $(e⁻ \\cdot s^{-1})$")
    plt.grid(True)
    plt.legend()
    plt.title("Dark current")

    plt.savefig(f"{path}Dark Current.svg")



'''
-------------------------------------------------------------------------------
'''
if __name__ == '__main__':
    dm = DM.DataManager()
    parser = argparse.ArgumentParser('PTC analysis')
    parser.add_argument('--camera', '-c', help='Select the cameras folder')
    parser.add_argument('--list', '-l', help='List all available cameras')
    args = parser.parse_args()

    inPath = "/data/" + args.camera + "/"
    outPath = "/img/" + args.camera + "/"
    outPath = dm.create_Dir(outPath)

    bias_df, dark_df, flat_df = load_datasets(dm, path=inPath)

    master_bias = build_master_bias(bias_df)
    center = [len(master_bias)//2, len(master_bias[0])//2]
    dx = 125
    dy = 125

    read_noise_dn = compute_read_noise(bias_df, center, dx, dy)

    ptc_df = compute_ptc(flat_df, master_bias, center, dx, dy)
    ptc_df = add_shot_noise(ptc_df, read_noise_dn)

    gain = fit_gain(ptc_df)
    prnu = compute_prnu(flat_df, master_bias, center, dx, dy)
    print(f"\nRead noise: {read_noise_dn} \nPRNU: {prnu} \nGain: {gain}")

    ptc_df = add_fpn(ptc_df, prnu)
    ptc_df = add_totalNoise(ptc_df)
    plot_ptc(ptc_df, read_noise_dn, path=outPath)

    plot_nonLinearity(ptc_df, path=outPath)

    dark_df = compute_dark_current(dark_df, master_bias, gain, center, dx, dy)
    plot_dark_current(dark_df, path=outPath)

    plt.show()


