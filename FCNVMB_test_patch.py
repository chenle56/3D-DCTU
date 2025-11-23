from ParamConfig import *
from PathConfig import *
from LibConfig import *
from torch.utils.data import DataLoader, TensorDataset

################################################
########         LOAD    NETWORK        ########
################################################

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()

device = torch.device("cuda" if cuda_available else "cpu")
model_file = models_dir + modelname + 'model_best.pkl'

net = SCUNet()

net.load_state_dict(torch.load(model_file))
print('model=',model_file)
net.to(device)
################################################
########    LOADING TESTING DATA       ########
################################################
print('***************** Loading Testing DataSet *****************')

test_set, label_set, data_dsp_dim, label_dsp_dim = DataLoad_Test(test_size = TestSize, test_data_dir = test_data_dir, \
                                                                 data_dim = DataDim, in_channels = Inchannels, \
                                                                 model_dim = ModelDim, data_dsp_blk = data_dsp_blk, \
                                                                 label_dsp_blk = label_dsp_blk, start = 1, \
                                                                 datafilename = datafilename, dataname = dataname, \
                                                                 truthfilename = truthfilename, truthname = truthname)

test = TensorDataset(torch.from_numpy(test_set), torch.from_numpy(label_set))
test_loader = DataLoader(test, batch_size = TestBatchSize, shuffle = False, drop_last = True)
################################################
########            TESTING             ########
################################################

print()
print('*******************************************')
print('*******************************************')
print('            START TESTING                  ')
print('*******************************************')
print('*******************************************')
print()

# Initialization
since = time.time()
TotPSNR = np.zeros((1, TestSize), dtype = float)
TotSSIM = np.zeros((1, TestSize), dtype = float)
Prediction = np.zeros((TestSize, label_dsp_dim[0], label_dsp_dim[1], label_dsp_dim[2]), dtype = 'float32')
GT = np.zeros((TestSize,  label_dsp_dim[0], label_dsp_dim[1], label_dsp_dim[2]), dtype = 'float32')
print('Prediction',Prediction.shape)
total = 0
predict_list = []
label_list=[]
noise_img_list=[]
count_img = 0
slide_window = 64
step_length = 32
pad_size = step_length // 2
patch_size = 64
stride = 64
sum_snr=0

for i, (images, labels) in enumerate(test_loader):
    print('***********************')
    print('output_shape', images.shape)
    print('labels_shape',labels.shape)
    images = images.view(TestBatchSize, 1, Inchannels, patch_size, patch_size)
    labels = labels.view(TestBatchSize, 1, Nclasses, patch_size, patch_size)
    images = images.to(device)
    labels = labels.to(device)

    # Predictions
    net.eval()
    outputs = net(images)

    outputs = outputs.reshape((TestBatchSize,Inchannels, patch_size, patch_size))
    print('output_shape', outputs.shape)
    labels=labels.reshape((TestBatchSize,Inchannels, patch_size, patch_size))
    print('labels_shape',labels.shape)
    noise_img=images.reshape((TestBatchSize,Inchannels, patch_size, patch_size))

    outputs = outputs.data.cpu().numpy()
    gts = labels.data.cpu().numpy()
    noise_imgs_slide=noise_img.data.cpu().numpy()

    # 3、Perform reasoning on the cropped small image above and load the model
    for k in range(TestBatchSize):
        predict_list.append(outputs[k,:, :, :])
        label_list.append(gts[k,:, :, :])
        noise_img_list.append(noise_imgs_slide[k,:, :, :])
        count_img += 1

print('count_img=',count_img)
count_temp = 0
sum_snr = 0
for k in range(TestSize):
    # 4.Simple concatenation only
    depth, height, width = label_dsp_dim[0], label_dsp_dim[1], label_dsp_dim[2]
    result_mask, gts_mask, imgs_mask = np.zeros(shape=(depth, height, width), dtype=np.float32), np.zeros(
        shape=(depth, height, width), dtype=np.float32), np.zeros(shape=(depth, height, width), dtype=np.float32)
    print('depth=',depth,height,width)
    for d in range(0, depth - Nclasses + 1, Nclasses):
        for i in range(0, height - patch_size + 1, stride):
            for j in range(0, width - patch_size + 1, stride):
                result_mask[d:d + Nclasses, i:i + patch_size, j:j + patch_size] = predict_list[count_temp][:, :, :]
                gts_mask[d:d + Nclasses, i:i + patch_size, j:j + patch_size] = label_list[count_temp][:, :, :]
                imgs_mask[d:d + Nclasses, i:i + patch_size, j:j + patch_size] = noise_img_list[count_temp][:, :, :]
                count_temp += 1

    snr = SNR(imgs_mask - result_mask, gts_mask)
    # Prediction[k,:, :, :] = turn(result_mask)
    # GT[k,:, :, :] = turn(gts_mask)
    Prediction[k,:, :, :] =imgs_mask - result_mask
    GT[k,:, :, :] = gts_mask
    # psnr = PSNR(result_mask,gts_mask)

    # print('gts_mask=',np.array(gts_mask).shape)
    # clip=0.1
    # plt.figure()
    # transposed_arr = np.transpose(Prediction.squeeze(), (1, 2, 0))
    # ima=transposed_arr[:,:,15]
    # plt.imshow(ima.squeeze(),vmin=-clip,vmax=clip,aspect='auto',cmap='viridis')
    # plt.show()

    TotPSNR[0, total] = snr
    sum_snr = snr + sum_snr
    ssim=0
    # ssim = SSIM(result_mask.reshape(-1, 1,Nclasses, label_dsp_dim[0], label_dsp_dim[1]),
    #             gts_mask.reshape(-1, 1,Nclasses, label_dsp_dim[0], label_dsp_dim[1]))
    TotSSIM[0, total] = ssim
    print('The %d testing snr: %.2f, SSIM: %.4f ' % (total, psnr, ssim))
    total = total + 1

print('The testset all-snr: %.2f ' % (sum_snr / total))

# Save Results
SaveTestResults(TotPSNR, TotSSIM,Prediction, GT, results_dir)

# Plot one prediction and ground truth
num = 0
if SimulateData:
    minvalue = 2000
else:
    minvalue = 1500
maxvalue = 4500
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 17,
         }
font3 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 21,
         }

PlotComparison(Prediction[0,0,:,:], GT[0,0,:,:], label_dsp_dim, label_dsp_blk, dh, minvalue, maxvalue, font2, font3,
               SavePath = results_dir)

# Record the consuming time
time_elapsed = time.time() - since
print('Testing complete in  {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
