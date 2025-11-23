SimulateData  = True          # If False denotes training the CNN with SEGSaltData
ReUse         = False         # If False always re-train a network 
DataDim       = [320,192]    # Dimension of original one-shot seismic data
data_dsp_blk  = (1,1)         # Downsampling ratio of input
ModelDim      = [320,192]       # Dimension of one velocity model
label_dsp_blk = (1,1)         # Downsampling ratio of output
dh            = 10            # Space interval


####################################################
####             NETWORK PARAMETERS             ####
####################################################
if SimulateData:
    Epochs        =30       # Number of epoch
    #TrainSize     = 1600      # Number of training set
    #TrainSize = 1600  # Number of training set
    TrainSize = 250  # Number of training set
    TestSize      = 25       # Number of testing set
    TestBatchSize = 1
else:
    Epochs        = 50
    TrainSize     = 130      
    TestSize      = 1
    TestBatchSize = 1
    
#BatchSize         = 10        # Number of batch size
BatchSize         = 1       # Number of batch size
LearnRate         = 2e-5   # Learning rate
Nclasses          = 64         # Number of output channels
#Inchannels        = 29        # Number of input channels, i.e. the number of shots
Inchannels        = 64        # Number of input channels, i.e. the number of shots
SaveEpoch         = 5
DisplayStep       = 2         # Number of steps till outputting stats
Patch_size        = 64
