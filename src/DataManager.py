import os
import pandas as pd
import numpy as np
from astropy.io import fits

ROOT = os.path.dirname(os.path.realpath(__file__))

class DataManager():
    def __init__(self):
        return None


    #Clear
    def clear(self):
        self.Dark.drop(labels=self.Dark.index, axis=0, inplace=True)
        self.linearity.drop(labels=self.linearity.index, axis=0, inplace=True)
        self.Gain.drop(labels=self.Gain.index, axis=0, inplace=True)
        #self.Noise.drop(labels=self.Noise.index, axis=0, inplace=True)

        
    #Stacks the Images inside the Frames
    def stack(self, frame):
        stacked_Frame = np.median(np.stack(frame.iloc[:, 1].to_numpy()), axis=0)
        return stacked_Frame

       
    def openFit(self, path, fileName):
        with fits.open(path+fileName) as fit:
            data = fit[0].data
        return data

    
    #Sorts files' data into corresponding DataFrames
    def fetchFiles(self, path, name_structure=["type", "Nr"], extension="fit"):
        path = ROOT + path
        files = [file for file in os.listdir(path) if file.endswith(extension)]
        parsed = []

        for file in files:
            parts = file.replace(f".{extension}", "").split("_")
            
            entry = {}
            for i, key in enumerate(name_structure):
                try:
                    entry[key] = float(parts[i]) 
                except ValueError:
                    entry[key] = parts[i].lower()
                except IndexError:
                    entry[key] = None
            entry["file"] = file
            entry["data"] = self.openFit(path, file)
            parsed.append(entry)

        df = pd.DataFrame(parsed)
        df.sort_values(name_structure, inplace=True)
        return df



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
        slice = image[y_min:y_max, x_min:x_max]
        y, x = np.ogrid[y_min:y_max, x_min:x_max]
        area = (x - center[0]) * (y - center[1])
        mask = area <= 4*dx*dy

            #Apply the mask
        masked = np.where(mask, slice, np.nan)
        return masked
    

