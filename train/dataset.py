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
fixedAirfoilNormalization = True
# global switch, make data dimensionless?
makeDimLess = True
# global switch, remove constant offsets from pressure channel?
removePOffset = True

verbose = False

# Norm Switches
L2_norm_switch = True
L1_norm_switch = False
Linf_norm = False

L075_norm = False
L050_norm = False
L025_norm = False

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
            if i < 3 and verbose:
                print("")
                print(file)
                
                print("---------")
                print("Inputs")
                print ("Min Max [0] -- Stream_X")
                print(d[0].min())
                print(d[0].max())
                print("")

                print ("Min Max [1] -- Stream_Y")
                print(d[1].min())
                print(d[1].max())
                print("")

                print ("Min Max [2] -- Mask")
                print(d[2].min())
                print(d[2].max())
                print("")

                print("---------")
                print("Targets")
                print ("Min Max Mean NoOff [0] -- Pressure")
                print(d[3].min())
                print(d[3].max())
                print(np.mean(d[3]))
                print("")
                print(d[3].min() - np.mean(d[3]))
                print(d[3].max() - np.mean(d[3]))
                print("")

                print ("Min Max Mean [1] -- Vel_X")
                print(d[4].min())
                print(d[4].max())
                print(np.mean(d[4]))
                print("")

                print ("Min Max Mean [2] -- Vel_Y")
                print(d[5].min())
                print(d[5].max())
                print(np.mean(d[5]))
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
            # if verbose and i < 5:
            #     print("Pressure Mean [" + str(i) + "]: " + str(np.mean(data.targets[i,0,:,:])))

    # make dimensionless based on current data set
    if makeDimLess:
        temp_L1 = 0
        temp_L2 = 0
        temp_Linf = 0
        temp_L075 = 0
        temp_L050 = 0
        temp_L025 = 0

        temp_0_mean = 0
        temp_1_mean = 0

        for i in range(data.totalLength):
            # only scale outputs, inputs are scaled by max only
            absmax_x = np.max(np.abs(data.inputs[i,0,:,:]))
            absmax_y = np.max(np.abs(data.inputs[i,1,:,:]))

            # Norms
            L2_norm = (absmax_x**2 + absmax_y**2)**0.5
            L1_norm = absmax_x + absmax_y
            Linf_norm = max(absmax_x, absmax_y)

            L075_norm = (absmax_x**0.75 + absmax_y**0.75)**(4.0 / 3.0)
            L050_norm = (absmax_x**0.5 + absmax_y**0.5)**2
            L025_norm = (absmax_x**0.25 + absmax_y**0.25)**4

            # Active Norm, default L2
            v_norm = L2_norm

            if L2_norm_switch:
                v_norm = L2_norm
            if L1_norm_switch:
                v_norm = L1_norm
            if Linf_norm:
                v_norm = Linf_norm
            if L075_norm:
                v_norm = L075_norm
            if L050_norm:
                v_norm = L050_norm
            if L025_norm:
                v_norm = L025_norm

            tmp_0_mean = np.mean(data.inputs[i,0,:,:])
            tmp_1_mean = np.mean(data.inputs[i,1,:,:])

            if verbose and i < 3:
                print("")
                print("Sample " + str(i))
                print("L inf Norm")
                print(round(Linf_norm, 2))
                print("-------")
                print("L2 Norm")
                print(round(L2_norm, 2))
                print("-------")
                print("L1 Norm")
                print(round(L1_norm, 2))
                print("-------")
                print("L0.75 Norm")
                print(round(L075_norm, 2))
                print("-------") 
                print("L0.5 Norm")
                print(round(L050_norm, 2))
                print("-------")
                print("L0.25 Norm")
                print(round(L025_norm, 2))
                print("-------")

            temp_L2 += L2_norm
            temp_L1 += L1_norm
            temp_Linf += Linf_norm
            temp_L075 += L075_norm
            temp_L050 += L050_norm
            temp_L025 += L025_norm

            temp_0_mean += tmp_0_mean
            temp_1_mean += tmp_1_mean

            data.targets[i,0,:,:] /= v_norm**2
            data.targets[i,1,:,:] /= v_norm
            data.targets[i,2,:,:] /= v_norm

            if verbose and i < 3:
                min0 = np.min(data.targets[i,0,:,:])
                max0 = np.max(data.targets[i,0,:,:])

                min1 = np.min(data.targets[i,1,:,:])
                max1 = np.max(data.targets[i,1,:,:])

                min2 = np.min(data.targets[i,2,:,:])
                max2 = np.max(data.targets[i,2,:,:])

                print("Dimless Interval [0]")
                print("[{}, {}]".format(round(min0, 2), round(max0, 2)))
                print("Dimless Interval [1]")
                print("[{}, {}]".format(round(min1, 2), round(max1, 2)))
                print("Dimless Interval [2]")
                print("[{}, {}]".format(round(min2, 2), round(max2, 2)))
                print("")

        if verbose:
            print("")
            print("Mean L1 Norm:")
            print(round(temp_L1 / data.totalLength, 2))

            print("Mean L2 Norm:")
            print(round(temp_L2 / data.totalLength, 2))

            print("Mean L inf Norm:")
            print(round(temp_Linf / data.totalLength, 2))

            print("Mean L0.75 Norm:")
            print(round(temp_L075 / data.totalLength, 2))

            print("Mean L0.5 Norm:")
            print(round(temp_L050 / data.totalLength, 2))

            print("Mean L0.25 Norm:")
            print(round(temp_L025 / data.totalLength, 2))

            print("Mean [0]:")
            print(round(temp_0_mean / data.totalLength, 2))

            print("Mean [1]:")
            print(round(temp_1_mean / data.totalLength, 2))
            print("")

    # normalize to -1..1 range, from min/max of predefined
    if verbose:
        print("Normalizing to [-1, 1]")
    if fixedAirfoilNormalization:
        # hard coded maxima , inputs dont change
        if verbose:
            print("Hard coded maxima")
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
            print("Max values from loaded data")
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

    if verbose:
        for i in range(3):
            print("")
            print("Sample " + str(i) + " - Target")

            min0 = np.min(data.targets[i,0,:,:])
            max0 = np.max(data.targets[i,0,:,:])

            min1 = np.min(data.targets[i,1,:,:])
            max1 = np.max(data.targets[i,1,:,:])

            min2 = np.min(data.targets[i,2,:,:])
            max2 = np.max(data.targets[i,2,:,:])

            print("Clamped Interval [0]")
            print("[{}, {}]".format(round(min0, 5), round(max0, 5)))
            print("Clamped Interval [1]")
            print("[{}, {}]".format(round(min1, 5), round(max1, 5)))
            print("Clamped Interval [2]")
            print("[{}, {}]".format(round(min2, 5), round(max2, 5)))
            print("")

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
                absmax_x = np.max(np.abs(data.inputs[i,0,:,:]))
                absmax_y = np.max(np.abs(data.inputs[i,1,:,:]))

                # Norms
                L2_norm = (absmax_x**2 + absmax_y**2)**0.5
                L1_norm = absmax_x + absmax_y
                Linf_norm = max(absmax_x, absmax_y)

                L075_norm = (absmax_x**0.75 + absmax_y**0.75)**(4.0 / 3.0)
                L050_norm = (absmax_x**0.5 + absmax_y**0.5)**2
                L025_norm = (absmax_x**0.25 + absmax_y**0.25)**4

                # Active Norm, default L2
                v_norm = L2_norm

                if L2_norm_switch:
                    v_norm = L2_norm
                if L1_norm_switch:
                    v_norm = L1_norm
                if Linf_norm:
                    v_norm = Linf_norm
                if L075_norm:
                    v_norm = L075_norm
                if L050_norm:
                    v_norm = L050_norm
                if L025_norm:
                    v_norm = L025_norm

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

        if L2_norm_switch:
            print("L2 Regularization")
        if L1_norm_switch:
            print("L1 Regularization")
        if Linf_norm:
            print("L_inf Regularization")
        if L075_norm:
            print("L0.75 Regularization")
        if L050_norm:
            print("L0.5 Regularization")
        if L025_norm:
            print("L0.25 Regularization")

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

