import os
import pandas as pd
import numpy as np
from astropy.io import fits

ROOT = os.path.dirname(os.path.realpath(__file__))

class DataManager():
    def __init__(self):
        #Init Dataframe structure
        self.df = pd.DataFrame(columns=[
            "files"
        ])

        
    #Stacks the Images inside the Frames
    def stack(self, frame):
        stacked_Frame = np.median(np.stack(frame.iloc[:, 1].to_numpy()), axis=0)
        return stacked_Frame

       
    def openFit(self, path, fileName):
        with fits.open(path+fileName) as fit:
            header = fit[0].header
            data = fit[0].data
            exp = header['EXPTIME']
            temp = header['CCD-TEMP']
        return pd.Series([data, exp, temp])

    
    def fetchFiles(self, path, extension="fit"):
        path = ROOT + path
        files = [f for f in os.listdir(path)] # if f.endswith(extension)] TODO Multi format readin
        self.df['files'] = files
        self.df['split_names'] = self.df['files'].apply(lambda n: n.replace(f".{extension}", "").split("_"))
        self.df[['data', 'exp', 'temp']] = self.df['files'].apply(lambda f: self.openFit(path, f))
        
        return self.df
    

    '''
    @Params:
        path.. project root path to output folder
    @return:
        path.. global path to output folder
    '''
    def createDir(self, path):
        path = ROOT + path
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            print(f"Error creating folder '{path}': {e}")
        return path


    '''
    @Params
        path.. project root path to input storage
    @return
        arr.. array with all the subdirs
    '''
    def getDirContent(self, path):
        path = ROOT + path
        folders = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
        return folders


    '''
    @Param:
        image.. the image where to apply the mask on
        center.. pixels [x, y] of mask origin
        radius.. mask size in pixels
    '''
    def circularSelection(self, image, centers, radii):
        fluxes = []

        for center, radius in zip(centers, radii):
            y_min = center[1]-radius
            y_max = center[1]+radius
            x_min = center[0]-radius
            x_max = center[0]+radius

            # Extract the rectangular region
            slice = image[y_min:y_max, x_min:x_max]

            # Create a circular mask for this region
            y, x = np.ogrid[y_min:y_max, x_min:x_max]
            distance = (x - center[0])**2 + (y - center[1])**2
            mask = distance <= radius**2

            #Apply the mask
            masked = np.where(mask, slice, np.nan)
            fluxes.append(np.nansum(masked))
        
        return fluxes


    '''
    @Param:
        image.. the image where to apply the mask on
        center.. pixels [x, y] of mask origin
        dx.. half mask width in pixels
        dy.. half mask height in pixels
    '''
    def rectangularSelection(self, image, center, dx, dy):

        y_min = center[1]-dy
        y_max = center[1]+dy
        x_min = center[0]-dx
        x_max = center[0]+dx

        # Extract the rectangular region
        return image[y_min:y_max, x_min:x_max]

        '''
        y, x = np.ogrid[y_min:y_max, x_min:x_max]
        area = x * y
        mask = area <= dx*dy
        
        
        h, w = slice.shape 
        yy, xx = np.ogrid[:h, :w]
        cy = center[1] - y_min
        cx = center[0] - x_min
        mask = ((xx - cx) * (yy - cy)) <= 4 * dx * dy
        

        #Apply the mask
        masked = np.where(mask, slice, np.nan)

        return masked
        '''
    

