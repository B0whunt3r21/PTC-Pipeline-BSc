import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd
import os
import argparse
from rich.console import Console
from rich.table import Table
from lmfit  import Model, models, report_fit, Parameter, printfuncs

import DataManager as DM



'''
    @Params:
        dm.. DataManager Module
        path.. relative file path
    @return
        bias_df.. bias dataframe
        dark_df.. dark dataframe
        light_df.. flat field dataframe
'''
def load_datasets(dm, path):
    """
    Reads all FITS files using your DataManager and returns:
    - bias_df
    - dark_df
    - light_df
    """
    df = dm.fetchFiles(path)
    df["type"] = df["split_names"].apply(lambda x: x[0])

    bias_df = df[df["type"] == "bias"].copy()

    dark_df = df[df["type"] == "dark"].copy()
    dark_df = dark_df.sort_values(["temp"])

    light_df = df[df["type"] == "light"].copy()
    light_df = light_df.sort_values(["exp"])

    return bias_df, dark_df, light_df


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
        read_noise.. read noise from bias differentiating
'''
def compute_read_noise(bias_df, center, dx, dy):
    b1 = dm.rectangularSelection(bias_df.iloc[0]["data"], center, dx, dy)
    b2 = dm.rectangularSelection(bias_df.iloc[1]["data"], center, dx, dy)
    return np.std(constantDifferentiating(b1, b2)/np.sqrt(2))


'''
    @Params:
        light_df.. flat data
        master_bias.. stacked master bias
        center.. [x, y] central point of mask
        dx.. half width of mask
        dy.. half heigth of mask
    @return
        ptc_df.. Basic PTC data
'''
def compute_ptc(light_df, master_bias, center, dx, dy):
    rows = []

    for exp, group in light_df.groupby("exp"):
        I1 = group["data"].iloc[0] - master_bias
        I2 = group["data"].iloc[-1] - master_bias

        roi1 = dm.rectangularSelection(I1, center, dx, dy)
        roi2 = dm.rectangularSelection(I2, center, dx, dy)

        signal = roi1.mean()
        spatial_noise = roi1.std()
        delta_noise = np.std(roi1 - roi2)/np.sqrt(2)

        rows.append({
            "exp": exp,
            "signal": signal,
            "spatial_noise": spatial_noise,
            "delta_noise": delta_noise
        })

    return pd.DataFrame(rows).sort_values("exp")


'''
    @Params:
        ptc_df.. PTC Data
    @return
        fwc.. full well capacity
'''
def get_FWC(ptc_df, percent=1):
    return ptc_df["signal"].iloc[ptc_df['delta_noise'].idxmax()+1] * percent


'''
    @Params:
        ptc_df.. PTC Data
        read_noise.. read noise in ADU/DN
    @return
        ptc_df.. PTC Dataframe with shoit noise column
'''
def add_shot_noise(ptc_df, read_noise):
    ptc_df["shot_noise"] = np.sqrt(ptc_df["delta_noise"]**2 - read_noise**2)
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
    idmax = ptc_df["shot_noise"].idxmax()+1
    x = ptc_df["signal"].iloc[:idmax]
    y = ptc_df["shot_noise"].iloc[:idmax]**2
    slope, intercept = np.polyfit(x, y, 1)
    gain = 1 / slope
    return gain


'''
    @Params:
        light_df.. Flat data
        master_bias.. stacked master bias
        center.. [x, y] central point of mask
        dx.. half width of mask
        dy.. half heigth of mask
    @return
        prnu.. calculated Pixel response non-unifomrity
'''
def compute_prnu(ptc_df, fwc):
    low = 0.1 * fwc
    high = 0.8 * fwc
    mask = (ptc_df["signal"] >= low) & (ptc_df["signal"] <= high)
    df = ptc_df[mask]
    prnu, _ = np.polyfit(df["signal"], df["spatial_noise"], 1)


    '''fitModel = models.LinearModel()

    idx_max = ptc_df["spatial_noise"].idxmax()
    x = ptc_df["signal"].iloc[:idx_max+1]
    y = ptc_df["spatial_noise"].iloc[:idx_max+1]
    params = fitModel.guess(y, x=x)
    result = fitModel.fit(y, params, x=x, calc_covar=True)
    prnu = result.params['slope'] / ptc_df['spatial_noise'].mean()
    print(prnu)
    '''

    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    ax.plot(ptc_df['signal'], ptc_df['spatial_noise'])
    #ax.plot(x, result.best_fit)
    ax.plot(df["signal"], prnu*df["signal"]+_)
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
def plot_ptc(ptc_df, read_noise, fwc, path):
    fitModel = models.LinearModel()

    idx_max = ptc_df["fpn"].idxmax()+1
    xFPN = ptc_df["signal"].iloc[:idx_max]
    yFPN = ptc_df["fpn"].iloc[:idx_max]
    fpn_params = fitModel.guess(yFPN, x=xFPN)
    fpn_result = fitModel.fit(yFPN, fpn_params, x=xFPN, calc_covar=True)


    idx_max = ptc_df["shot_noise"].idxmax()+1
    xShot = ptc_df["signal"].iloc[:idx_max]
    yShot = ptc_df["shot_noise"].iloc[:idx_max]**2
    shot_params = fitModel.make_params(slope=0.5, intercept=0)
    shot_params['intercept'].vary = False
    shot_result = fitModel.fit(yShot, shot_params, x=xShot, calc_covar=True)


    plt.figure(figsize=(6, 6))
    plt.loglog(ptc_df["signal"], ptc_df["delta_noise"], color=color_Delta_Noise, marker="s", linestyle=":", label="Temporal noise")
    
    plt.hlines(read_noise, 0, ptc_df["signal"].min(), color=color_Read_Noise, linestyle='-', label="Read noise")
    
    plt.loglog(ptc_df["signal"], ptc_df["shot_noise"], color=color_Shot_Noise, marker="D", linestyle="", label="Shot noise")
    plt.loglog(xShot, np.sqrt(shot_result.best_fit), color=color_Shot_Noise, linestyle='--', label='shot noise best fit')
    
    plt.loglog(ptc_df["signal"], ptc_df["fpn"], color=color_FPN, marker="o", linestyle="", label="FPN")
    plt.loglog(xFPN, fpn_result.best_fit, color=color_FPN, linestyle='--', label='FPN best fit')
    
    plt.loglog(ptc_df["signal"], ptc_df["total_noise"], color=color_Total_Noise, marker="x", linestyle=":", label="Total noise")
    
    plt.vlines(fwc, 0, ptc_df["total_noise"].max(), color=color_FWC, linestyle='--', label="FWC")

    plt.xlim(1, 4e4)
    plt.ylim(1, 9e3)
    plt.xlabel("Signal (DN)")
    plt.ylabel("Noise (DN)")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.minorticks_on()
    plt.title("PTC")

    plt.savefig(f"{path}PTC.svg")


'''
    @Params:
        ptc_df.. PTC Data
        path.. root path to output folder
'''
def plot_nonLinearity(ptc_df, path, fwc):
    fitModel = models.LinearModel()

    idx_max = ptc_df["signal"].idxmax()
    x = ptc_df["exp"].iloc[:idx_max-1]
    y = ptc_df["signal"].iloc[:idx_max-1]
    params = fitModel.guess(y, x=x)
    result = fitModel.fit(y, params, x=x, calc_covar=True)
    
    xAx = ptc_df["exp"].iloc[:idx_max+1]
    yAx = result.eval(x=ptc_df["exp"].iloc[:idx_max+1])

    nonLin = np.abs(ptc_df["signal"].iloc[:idx_max+1] - yAx)/yAx.max() * 100
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 7))
    fig.suptitle("Nonlinearity")

    axes[0].plot(ptc_df["exp"], ptc_df["signal"], "bx--", label="Signals")
    axes[0].plot(xAx, yAx, color=color_Fit, linestyle='--', label='best fit')
    axes[0].hlines(fwc, ptc_df['exp'].iloc[0], ptc_df['exp'].iloc[-1], color=color_FWC, linestyle="--", label="FWC")
    axes[0].set_xlabel("Exposure (s)")
    axes[0].set_ylabel("Signal (DN)")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].minorticks_on()


    fitModel = models.PolynomialModel()

    params = fitModel.guess(nonLin, x=xAx)
    result = fitModel.fit(nonLin, params, x=xAx, calc_covar=True)

    axes[1].plot(xAx, nonLin, 'bx')
    axes[1].plot(xAx, result.best_fit, color=color_Fit, linestyle='--')
    axes[1].set_xlabel("Exposure (s)")
    axes[1].set_ylabel("Non-linearity (%)")
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
        dark_current = mean_e / group["exp"].mean()

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
    fitModel = models.ExponentialModel()
    params = fitModel.guess(dc_df['dark_current'], x=dc_df["temp"])
    result = fitModel.fit(dc_df['dark_current'], params, x=dc_df["temp"], calc_covar=True)
    console.log("Dark current at 25°C: ", result.eval(x=25))

    plt.figure(figsize=(6, 4))
    plt.plot(dc_df["temp"], dc_df["dark_current"], "bx", label="Dark current")
    plt.plot(dc_df["temp"], result.best_fit, color=color_Fit, linestyle='--', label='best fit')
    plt.xlabel("Temperature $(°C)$")
    plt.ylabel("Dark current $(e⁻ \\cdot s^{-1})$")
    plt.grid(True)
    plt.legend()
    plt.title("Dark current")

    plt.savefig(f"{path}Dark Current.svg")



'''
_________________________________________________________________
|                    Statistics (Histograms)                    |
|_______________________________________________________________|
'''

'''
    @Prams:
        imageData.. row of image to make histogram of
'''
def makeHist(imageData):
    plt.figure(figsize=(6, 4))
    plt.hist(imageData['data'].flatten(), bins="auto")
    #plt.vlines([imageData['data'].mean() - imageData['data'].std(), imageData['data'].mean(), imageData['data'].mean() + imageData['data'].std()], 0, len(imageData['data'][0]), colors=['gray', 'red', 'gray'], linestyles=[':', '--', ':'])
    plt.xlabel("Value (ADU)")
    plt.ylabel("#Px")
    plt.grid(True)
    #plt.legend()
    plt.title(f"Histogram of {imageData['files']}")



'''
-------------------------------------------------------------------------------
'''
if __name__ == '__main__':
    dm = DM.DataManager()
    console = Console()
    parser = argparse.ArgumentParser('PTC analysis')
    parser.add_argument('--camera', '-c', help='Select the cameras folder')
    parser.add_argument('--list', '-l', action='store_true', help='List all available cameras')
    args = parser.parse_args()


    color_Read_Noise = "#000000"
    color_Delta_Noise = "#0072B2"
    color_Total_Noise = "#D55E00"
    color_FPN = "#009E73"
    color_Shot_Noise = "#E69F00"
    color_FWC = "#999999"
    color_DR = "#FFFFFF"
    color_Fit = "#CC00CC"


    if args.list:
        cams = dm.getDirContent(path="/data/")

        table = Table(title="Available Cameras")
        table.add_column("Name", style="cyan", justify="center")
        for cam in cams:
            table.add_row(cam)
        console.print(table)


    if args.camera == None:
        pass
    else:
        inPath = "/data/" + args.camera + "/"
        outPath = "/img/" + args.camera + "/"
        outPath = dm.createDir(outPath)

        bias_df, dark_df, light_df = load_datasets(dm, path=inPath)

        master_bias = build_master_bias(bias_df)
        center = [len(master_bias)//2, len(master_bias[0])//2]
        dx = 125
        dy = 125

        read_noise = compute_read_noise(bias_df, center, dx, dy)

        ptc_df = compute_ptc(light_df, master_bias, center, dx, dy)
        ptc_df = add_shot_noise(ptc_df, read_noise)

        fwc = get_FWC(ptc_df)

        gain = fit_gain(ptc_df)

        prnu = compute_prnu(ptc_df, fwc)
        ptc_df = add_fpn(ptc_df, prnu)

        ptc_df = add_totalNoise(ptc_df)

        
        plot_ptc(ptc_df, read_noise, fwc, path=outPath)


        dr = fwc/read_noise
        dr_db = 20 * np.log10(dr)


        plot_nonLinearity(ptc_df, outPath, fwc)

        dark_df = compute_dark_current(dark_df, master_bias, gain, center, dx, dy)
        plot_dark_current(dark_df, path=outPath)


        #Print PTC Results
        table = Table(title=f"PTC resuts for {args.camera}")
        table.add_column("Gain", style=color_Shot_Noise, justify="center")
        table.add_column("read noise", style=color_Read_Noise, justify="center")
        table.add_column("PRNU", style=color_FPN, justify="center")
        table.add_column("FWC", style=color_FWC, justify="center")
        table.add_column("DR", style=color_DR, justify="center")
        table.add_column("DR (DB)", style=color_DR, justify="center")
        table.add_row(str(round(gain, 4)), str(round(read_noise, 4)), str(round(prnu, 4)), str(round(fwc, 4)), str(round(dr, 4)), str(round(dr_db, 4)))
        console.print(table)


        makeHist(light_df.iloc[-1])


        #plt.show()

