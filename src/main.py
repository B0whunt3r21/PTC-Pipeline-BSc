import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from uncertainties import ufloat, nominal_value, unumpy
import numpy as np
import pandas as pd
import os

import DataManager as DM

dm = DM.DataManager()


'''
@Param:
    img1.. the image on which che constant gets added
    img2.. image te be substracted from img1
'''
def constantDifferentiating(img1, img2):
    const = (img1 + 10000) * 0.01
    diffIMG = img1 + const - img2
    return diffIMG



'''
@Param:
    totalData.. complete dataframe of the files
    gain.. cameras calculated gain
'''
def readoutNoise(totalData, gain):
    diff = constantDifferentiating(totalData['data'].loc[totalData['Nr'] == 1].to_numpy(), totalData['data'].loc[totalData['Nr'] == 2].to_numpy())
    subIMG = dm.rectangularSelection(diff[0], [len(diff[0])//2, len(diff[0][0])//2], 50, 50)
    sigma = np.std(subIMG)
    readout = sigma / np.sqrt(2) * gain
    print(f'\n\nReadout noise: {readout}\n\n')



'''
@Param:
    totalGainData.. complete dataframe of the files
'''
def gainCalculation(totalGainData):
    
    differenialGains = totalGainData.groupby('step')['data'].apply(lambda imgs: constantDifferentiating(imgs.iloc[0], imgs.iloc[1])).reset_index()

    differenialGains['means'] = None
    differenialGains['rms'] = None
    
    for i in range(len(differenialGains['data'])):
        differenialGains.at[i, "means"] = np.mean(differenialGains['data'].iloc[i])
        differenialGains.at[i, "rms"] = np.sqrt(np.mean(np.square(differenialGains['data'].iloc[i])))

    x = pd.to_numeric(differenialGains['means'], errors='coerce').to_numpy(dtype=float)
    y = pd.to_numeric(differenialGains['rms'], errors='coerce').to_numpy(dtype=float)

    fitX = np.arange(x.min(), x.max(), 100)
    coef, cov = np.polyfit(x, y, 1, cov=True)
    fit = np.poly1d(coef)

    k = ufloat(coef[0], np.sqrt(np.diag(cov))[1])
    d = ufloat(coef[1], np.sqrt(np.diag(cov))[1])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(differenialGains['means'], differenialGains['rms'], linestyle='', marker='x')
    ax.plot(fitX, fit(fitX), linestyle=':', marker='')
    ax.set_xlabel('Mean (ADU)')
    ax.set_ylabel('RMS (ADU)')

    print(f"\n\ngain: {k}\nordinate: {d}\n\n")
    return nominal_value(k)



'''
@Param:
    totalDarkData.. complete dataframe of the files
    gain.. cameras calculated gain
'''
def darkCurrent(totalDarkData, gain):
    
    darkSubset = totalDarkData[totalDarkData['Type'] == 'dark'].reset_index()
    biasSubset = totalDarkData[totalDarkData['Type'] == 'bias'].reset_index()

    stackedBiases = biasSubset.groupby('Temperature')['data'].apply(lambda imgs: np.median(np.stack(imgs.to_numpy()), axis=0)).reset_index()
    stackedDarks = darkSubset.groupby('Temperature')['data'].apply(lambda imgs: np.median(np.stack(imgs.to_numpy()), axis=0)).reset_index()

    stackedBiases['means'] = None
    stackedDarks['means'] = None
    stackedDarks['current'] = None

    for i in range(len(stackedDarks['data'])):
        stackedDarks.at[i, "means"] = np.mean(dm.rectangularSelection(stackedDarks['data'].iloc[i], [len(stackedDarks['data'].iloc[i])//2, len(stackedDarks['data'].iloc[i][0])//2], 50, 50))
        stackedBiases.at[i, "means"] = np.mean(dm.rectangularSelection(stackedBiases['data'].iloc[i], [len(stackedBiases['data'].iloc[i])//2, len(stackedBiases['data'].iloc[i][0])//2], 50, 50))

    stackedDarks['current'] = (stackedDarks['means'] - stackedBiases['means'])/60 * gain
    
    '''
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(darkSubset['data'].iloc[0])
    rect = Rectangle([len(stackedBiases['data'].iloc[i])//2, len(stackedBiases['data'].iloc[0][0])//2], 50, 50)
    ax.add_patch(rect)
    '''

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(stackedDarks['Temperature'], stackedDarks['current'])
    ax.set_xlabel('Temperature ($^\circ$C)')
    ax.set_ylabel('Dark Current (e-/s)')



'''
@Param:
    totalLinearityData.. complete dataframe of the files
    gain.. cameras calculated gain
'''
def linearityAnalysis(totalLinearityData, gain):
    
    darkSubset = totalLinearityData[totalLinearityData['Type'] == 'dark'].reset_index()
    biasSubset = totalLinearityData[totalLinearityData['Type'] == 'bias'].reset_index()
    flatSubset = totalLinearityData[totalLinearityData['Type'] == 'linearity'].reset_index()

    stackedBiases = biasSubset.groupby('Type')['data'].apply(lambda imgs: np.median(np.stack(imgs.to_numpy()), axis=0)).reset_index()
    
    flatSubset['means'] = None
    flatSubset['rms'] = None
    flatSubset['corr'] = flatSubset['data']*0

    for i in range(len(flatSubset['data'])):
        flatSubset.at[i, "corr"] = flatSubset['data'].iloc[i] - (((darkSubset['data'].iloc[0] / 60) - stackedBiases['data'].iloc[0]) * flatSubset['delim'].iloc[i])
        rect = dm.rectangularSelection(flatSubset['corr'].iloc[i], [len(flatSubset['corr'].iloc[i])//2, len(flatSubset['corr'].iloc[i][0])//2], 50, 50)
        flatSubset.at[i, "means"] = np.mean(rect)
        flatSubset.at[i, "rms"] = np.sqrt(np.mean(np.square(rect)))
        
    '''
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(totalLinearityData['data'].iloc[0])
    rect = Rectangle([len(flatSubset['data'].iloc[i])//2, len(flatSubset['data'].iloc[0][0])//2], 50, 50)
    ax.add_patch(rect)
    '''

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(flatSubset['delim'], flatSubset['means'] * gain, linestyle='', marker='x')
    ax.set_xlabel('Integration time (s)')
    ax.set_ylabel('counts')




if __name__ == '__main__':

    gainDF = dm.fetchFiles("/data/testShots/gain/", ['Type', 'step', 'Nr'])
    gain = gainCalculation(gainDF)

    darkDF = dm.fetchFiles("/data/testShots/dark/", ['Type', 'Temperature', 'Nr'])
    darkCurrent(darkDF, gain)

    fwcDF = dm.fetchFiles("/data/testShots/FWC/", ['Type', 'delim'])
    linearityAnalysis(fwcDF, gain)

    readNoise = dm.fetchFiles("/data/testShots/ReadNoise/", ['Type', 'Nr'])
    readoutNoise(readNoise, gain)

    plt.show()


