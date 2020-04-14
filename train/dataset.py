################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Dataset handling
#
################

from torch.utils.data import Dataset
import numpy as np
from os import listdir
import random

# global switch, use fixed max values for dim-less airfoil data?
fixedAirfoilNormalization = False
# global switch, make data dimensionless?
makeDimLess = True
# global switch, remove constant offsets from pressure channel?
removePOffset = True

verbose = True
L1L2switch = True

## helper - compute absolute of inputs or targets
def find_absmax(data, use_targets, x):
    maxval = 0
    for i in range(data.totalLength):
        if use_targets == 0:
            temp_tensor = data.inputs[i]
        else:
            temp_tensor = data.targets[i]
        temp_max = np.max(np.abs(temp_tensor[x]))
        if temp_max > maxval:
            maxval = temp_max
    return maxval


######################################## DATA LOADER #########################################
#         also normalizes data with max , and optionally makes it dimensionless              #

def LoaderNormalizer(data, isTest = False, shuffle = 0, dataProp = None):
    """
    # data: pass TurbDataset object with initialized dataDir / dataDirTest paths
    # train: when off, process as test data (first load regular for normalization if needed, then replace by test data)
    # dataProp: proportions for loading & mixing 3 different data directories "reg", "shear", "sup"
    #           should be array with [total-length, fraction-regular, fraction-superimposed, fraction-sheared],
    #           passing None means off, then loads from single directory
    """

    if dataProp is None:
        # load single directory
        files = listdir(data.dataDir)
        files.sort()
        for i in range(shuffle):
            random.shuffle(files) 
        if isTest:
            print("Reducing data to load for tests")
            files = files[0:min(10, len(files))]
        data.totalLength = len(files)
        data.inputs  = np.empty((len(files), 3, 128, 128))
        data.targets = np.empty((len(files), 3, 128, 128))

        for i, file in enumerate(files):
            npfile = np.load(data.dataDir + file)
            d = npfile['a']
            if i < 5 and verbose:
                print("")
                print(file)
                
                print("---------")
                print("Inputs")
                print ("Min Max Mean Clamp [0] -- Stream_X")
                print(d[0].min())
                print(d[0].max())
                print(np.mean(d[0]))
                print("")
                print(d[0].min() * (1 / np.max(np.abs(d[0]))))
                print(d[0].max() * (1 / np.max(np.abs(d[0]))))
                print("")

                print ("Min Max Mean Clamp [1] -- Stream_Y")
                print(d[1].min())
                print(d[1].max())
                print(np.mean(d[1]))
                print("")
                print(d[1].min() * (1 / np.max(np.abs(d[1]))))
                print(d[1].max() * (1 / np.max(np.abs(d[1]))))
                print("")

                print ("Min Max Mean [2] -- Mask")
                print(d[2].min())
                print(d[2].max())
                print(np.mean(d[2]))
                print("")

                print("---------")
                print("Targets")
                print ("Min Max Mean NoOff Norm Clamp ClampPredefined [0] -- Pressure")
                print(d[3].min())
                print(d[3].max())
                print(np.mean(d[3]))
                print("")
                print(d[3].min() - np.mean(d[3]))
                print(d[3].max() - np.mean(d[3]))
                print("")
                temp_min_target3 = (d[3].min() - np.mean(d[3])) / ( np.max(np.abs(d[0]))**2 + np.max(np.abs(d[1]))**2 )
                temp_max_target3 = (d[3].max() - np.mean(d[3])) / ( np.max(np.abs(d[0]))**2 + np.max(np.abs(d[1]))**2 )
                print(temp_min_target3)
                print(temp_max_target3)
                print("")
                print(temp_min_target3 * (1.0 / np.max(np.abs(np.array([temp_min_target3, temp_max_target3])))))
                print(temp_max_target3 * (1.0 / np.max(np.abs(np.array([temp_min_target3, temp_max_target3])))))
                print("")
                print(temp_min_target3 * (1.0 / 4.65))
                print(temp_max_target3 * (1.0 / 4.65))
                print("")

                print ("Min Max Mean Norm Clamp ClampPredefined [1] -- Vel_X")
                print(d[4].min())
                print(d[4].max())
                print(np.mean(d[4]))
                print("")
                temp_min_target4 = d[4].min() / ( np.max(np.abs(d[0]))**2 + np.max(np.abs(d[1]))**2 )**0.5
                temp_max_target4 = d[4].max() / ( np.max(np.abs(d[0]))**2 + np.max(np.abs(d[1]))**2 )**0.5
                print(temp_min_target4)
                print(temp_max_target4)
                print("")
                print(temp_min_target4 * (1.0 / np.max(np.abs(np.array([temp_min_target4, temp_max_target4])))))
                print(temp_max_target4 * (1.0 / np.max(np.abs(np.array([temp_min_target4, temp_max_target4])))))
                print("")
                print(temp_min_target4 * (1.0 / 2.04))
                print(temp_max_target4 * (1.0 / 2.04))
                print("")

                print ("Min Max Mean Norm Clamp ClampPredefined [2] -- Vel_Y")
                print(d[5].min())
                print(d[5].max())
                print(np.mean(d[5]))
                print("")
                temp_min_target5 = d[5].min() / ( np.max(np.abs(d[0]))**2 + np.max(np.abs(d[1]))**2 )**0.5
                temp_max_target5 = d[5].max() / ( np.max(np.abs(d[0]))**2 + np.max(np.abs(d[1]))**2 )**0.5
                print(temp_min_target5)
                print(temp_max_target5)
                print("")
                print(temp_min_target5 * (1.0 / np.max(np.abs(np.array([temp_min_target5, temp_max_target5])))))
                print(temp_max_target5 * (1.0 / np.max(np.abs(np.array([temp_min_target5, temp_max_target5])))))
                print("")
                print(temp_min_target5 * (1.0 / 2.04))
                print(temp_max_target5 * (1.0 / 2.04))
                print("")

            data.inputs[i] = d[0:3]
            data.targets[i] = d[3:6]
        print("Number of data loaded:", len(data.inputs) )

    else:
        # load from folders reg, sup, and shear under the folder dataDir
        data.totalLength = int(dataProp[0])
        data.inputs  = np.empty((data.totalLength, 3, 128, 128))
        data.targets = np.empty((data.totalLength, 3, 128, 128))

        files1 = listdir(data.dataDir + "reg/")
        files1.sort()
        files2 = listdir(data.dataDir + "sup/")
        files2.sort()
        files3 = listdir(data.dataDir + "shear/" )
        files3.sort()
        for i in range(shuffle):
            random.shuffle(files1) 
            random.shuffle(files2) 
            random.shuffle(files3) 

        temp_1, temp_2 = 0, 0
        for i in range(data.totalLength):
            if i >= (1-dataProp[3])*dataProp[0]:
                npfile = np.load(data.dataDir + "shear/" + files3[i-temp_2])
                d = npfile['a']
                data.inputs[i] = d[0:3]
                data.targets[i] = d[3:6]
            elif i >= (dataProp[1])*dataProp[0]:
                npfile = np.load(data.dataDir + "sup/" + files2[i-temp_1])
                d = npfile['a']
                data.inputs[i] = d[0:3]
                data.targets[i] = d[3:6]
                temp_2 = i + 1
            else:
                npfile = np.load(data.dataDir + "reg/" + files1[i])
                d = npfile['a']
                data.inputs[i] = d[0:3]
                data.targets[i] = d[3:6]
                temp_1 = i + 1
                temp_2 = i + 1
        print("Number of data loaded (reg, sup, shear):", temp_1, temp_2 - temp_1, i+1 - temp_2)

    ################################## NORMALIZATION OF TRAINING DATA ##########################################

    if removePOffset:
        if verbose:
            print("removePOffset - Targets")
        for i in range(data.totalLength):
            data.targets[i,0,:,:] -= np.mean(data.targets[i,0,:,:]) # remove offset
            data.targets[i,0,:,:] -= data.targets[i,0,:,:] * data.inputs[i,2,:,:]  # pressure * mask
            if verbose and i < 5:
                print("Pressure Mean [" + str(i) + "]: " + str(np.mean(data.targets[i,0,:,:])))

    # make dimensionless based on current data set
    if makeDimLess:
        if verbose:
            print("makeDimLess - Targets")
            print("L2 Norm")
            print((np.max(np.abs(data.inputs[i,0,:,:]))**2 + np.max(np.abs(data.inputs[i,1,:,:]))**2 )**0.5)
            print("-------")
            print("L1 Norm")
            print(np.max(np.abs(data.inputs[i,0,:,:])) + np.max(np.abs(data.inputs[i,1,:,:])))
        for i in range(data.totalLength):
            # only scale outputs, inputs are scaled by max only
            if L1L2switch:
                v_norm = ( np.max(np.abs(data.inputs[i,0,:,:]))**2 + np.max(np.abs(data.inputs[i,1,:,:]))**2 )**0.5
            else:
                v_norm = np.max(np.abs(data.inputs[i,0,:,:])) + np.max(np.abs(data.inputs[i,1,:,:]))
                
            data.targets[i,0,:,:] /= v_norm**2
            data.targets[i,1,:,:] /= v_norm
            data.targets[i,2,:,:] /= v_norm

    # normalize to -1..1 range, from min/max of predefined
    if fixedAirfoilNormalization:
        # hard coded maxima , inputs dont change
        data.max_inputs_0 = 100.
        data.max_inputs_1 = 38.12
        data.max_inputs_2 = 1.0

        # targets depend on normalization
        if makeDimLess:
            data.max_targets_0 = 4.65 
            data.max_targets_1 = 2.04
            data.max_targets_2 = 2.37
            print("Using fixed maxima "+format( [data.max_targets_0,data.max_targets_1,data.max_targets_2] ))
        else: # full range
            data.max_targets_0 = 40000.
            data.max_targets_1 = 200.
            data.max_targets_2 = 216.
            print("Using fixed maxima "+format( [data.max_targets_0,data.max_targets_1,data.max_targets_2] ))

    else: # use current max values from loaded data
        if verbose:
            print("fixed input maxima  [100, 38.12, 1.00] for comparison")
            print("fixed target maxima [4.65, 2.04, 2.37] for comparison")
        data.max_inputs_0 = find_absmax(data, 0, 0)
        data.max_inputs_1 = find_absmax(data, 0, 1)
        data.max_inputs_2 = find_absmax(data, 0, 2) # mask, not really necessary
        print("Maxima inputs "+format( [data.max_inputs_0,data.max_inputs_1,data.max_inputs_2] )) 

        data.max_targets_0 = find_absmax(data, 1, 0)
        data.max_targets_1 = find_absmax(data, 1, 1)
        data.max_targets_2 = find_absmax(data, 1, 2)
        print("Maxima targets "+format( [data.max_targets_0,data.max_targets_1,data.max_targets_2] )) 
    
    data.inputs[:,0,:,:] *= (1.0/data.max_inputs_0)
    data.inputs[:,1,:,:] *= (1.0/data.max_inputs_1)

    data.targets[:,0,:,:] *= (1.0/data.max_targets_0)
    data.targets[:,1,:,:] *= (1.0/data.max_targets_1)
    data.targets[:,2,:,:] *= (1.0/data.max_targets_2)

    ###################################### NORMALIZATION  OF TEST DATA #############################################

    if isTest:
        files = listdir(data.dataDirTest)
        files.sort()
        data.totalLength = len(files)
        data.inputs  = np.empty((len(files), 3, 128, 128))
        data.targets = np.empty((len(files), 3, 128, 128))
        for i, file in enumerate(files):
            npfile = np.load(data.dataDirTest + file)
            d = npfile['a']
            data.inputs[i] = d[0:3]
            data.targets[i] = d[3:6]

        if removePOffset:
            for i in range(data.totalLength):
                data.targets[i,0,:,:] -= np.mean(data.targets[i,0,:,:]) # remove offset
                data.targets[i,0,:,:] -= data.targets[i,0,:,:] * data.inputs[i,2,:,:]  # pressure * mask

        if makeDimLess:
            for i in range(len(files)):
                if L1L2switch:
                    v_norm = ( np.max(np.abs(data.inputs[i,0,:,:]))**2 + np.max(np.abs(data.inputs[i,1,:,:]))**2 )**0.5
                else:
                    v_norm = np.max(np.abs(data.inputs[i,0,:,:])) + np.max(np.abs(data.inputs[i,1,:,:]))
                data.targets[i,0,:,:] /= v_norm**2
                data.targets[i,1,:,:] /= v_norm
                data.targets[i,2,:,:] /= v_norm
    
        data.inputs[:,0,:,:] *= (1.0/data.max_inputs_0)
        data.inputs[:,1,:,:] *= (1.0/data.max_inputs_1)

        data.targets[:,0,:,:] *= (1.0/data.max_targets_0)
        data.targets[:,1,:,:] *= (1.0/data.max_targets_1)
        data.targets[:,2,:,:] *= (1.0/data.max_targets_2)

    print("Data stats, input  mean %f, max  %f;   targets mean %f , max %f " % ( 
      np.mean(np.abs(data.targets), keepdims=False), np.max(np.abs(data.targets), keepdims=False) , 
      np.mean(np.abs(data.inputs), keepdims=False) , np.max(np.abs(data.inputs), keepdims=False) ) ) 

    return data

######################################## DATA SET CLASS #########################################

class TurbDataset(Dataset):

    # mode "enum" , pass to mode param of TurbDataset (note, validation mode is not necessary anymore)
    TRAIN = 0
    TEST  = 2

    def __init__(self, dataProp=None, mode=TRAIN, dataDir="../data/train/", dataDirTest="../data/test/", shuffle=0, normMode=0):
        global makeDimLess, removePOffset
        """
        :param dataProp: for split&mix from multiple dirs, see LoaderNormalizer; None means off
        :param mode: TRAIN|TEST , toggle regular 80/20 split for training & validation data, or load test data
        :param dataDir: directory containing training data
        :param dataDirTest: second directory containing test data , needs training dir for normalization
        :param normMode: toggle normalization
        """
        if not (mode==self.TRAIN or mode==self.TEST):
            print("Error - TurbDataset invalid mode "+format(mode) ); exit(1)

        if normMode==1:	
            print("Warning - poff off!!")
            removePOffset = False
        if normMode==2:	
            print("Warning - poff and dimless off!!!")
            makeDimLess = False
            removePOffset = False

        self.mode = mode
        self.dataDir = dataDir
        self.dataDirTest = dataDirTest # only for mode==self.TEST
        if L1L2switch:
            print("L2 Normalization")
        else:
            print("L1 Normalization")
        # load & normalize data
        self = LoaderNormalizer(self, isTest=(mode==self.TEST), dataProp=dataProp, shuffle=shuffle)
        
        if not self.mode==self.TEST:
            # split for train/validation sets (80/20) , max 400
            targetLength = self.totalLength - min( int(self.totalLength*0.2) , 400)

            self.valiInputs = self.inputs[targetLength:]
            self.valiTargets = self.targets[targetLength:]
            self.valiLength = self.totalLength - targetLength

            self.inputs = self.inputs[:targetLength]
            self.targets = self.targets[:targetLength]
            self.totalLength = self.inputs.shape[0]
        
    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    #  reverts normalization 
    def denormalize(self, data, v_norm):
        a = data.copy()
        a[0,:,:] /= (1.0/self.max_targets_0)
        a[1,:,:] /= (1.0/self.max_targets_1)
        a[2,:,:] /= (1.0/self.max_targets_2)

        if makeDimLess:
            a[0,:,:] *= v_norm**2
            a[1,:,:] *= v_norm
            a[2,:,:] *= v_norm
        return a

# simplified validation data set (main one is TurbDataset above)

class ValiDataset(TurbDataset):
    def __init__(self, dataset): 
        self.inputs = dataset.valiInputs
        self.targets = dataset.valiTargets
        self.totalLength = dataset.valiLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

