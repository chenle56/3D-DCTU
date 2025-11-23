from torch.utils.data import DataLoader, TensorDataset

from ParamConfig import *
from PathConfig import *
from LibConfig import *


################################################
########             NETWORK            ########
################################################

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

# net = UnetModel(n_classes=Nclasses,in_channels=Inchannels,is_deconv=True,is_batchnorm=True)
# net = DnCNN3d()
net = SCUNet()
if torch.cuda.is_available():
    net.cuda()

# Optimizer we want to use
optimizer = torch.optim.Adam(net.parameters(), lr=LearnRate)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
scaler = StandardScaler()

# ######### Scheduler ###########
# warmup_epochs = 3
# scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Epochs-warmup_epochs+40, eta_min=0)
# scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

# torch.optim.AdamW(net.parameters(),lr)
# type="StepLR",step_size=5,gamma=1/3


# If ReUse, it will load saved model from premodelfilepath and continue to train
if ReUse:
    print('***************** Loading the pre-trained model *****************')
    print('')
    premodel_file = models_dir + premodelname + '.pkl'
    ##Load generator parameters
    net.load_state_dict(torch.load(premodel_file))
    net = net.to(device)
    print('Finish downloading:', str(premodel_file))

################################################
########    LOADING TRAINING DATA       ########
################################################
print('***************** Loading Training DataSet *****************')
train_set, label_set, data_dsp_dim, label_dsp_dim = DataLoad_Train(train_size=TrainSize, train_data_dir=train_data_dir, \
                                                                   data_dim=DataDim, in_channels=Inchannels, \
                                                                   model_dim=ModelDim, data_dsp_blk=data_dsp_blk, \
                                                                   label_dsp_blk=label_dsp_blk, start=1, \
                                                                   datafilename=datafilename, dataname=dataname, \
                                                                   truthfilename=truthfilename, truthname=truthname,
                                                                   batchSize=BatchSize)
# Change data type (numpy --> tensor)
full_dataset = data_utils.TensorDataset(torch.from_numpy(train_set), torch.from_numpy(label_set))

val_ratio = 0.05  # 验证集比例
train_ratio = 0.45  # 训练集比例 (可以调整这个值来做对比实验)

valdata_size = int(len(full_dataset) * val_ratio)
# train_size = len(full_dataset) - valdata_size
train_size = int(len(full_dataset) * train_ratio)
remaining_size = len(full_dataset) - valdata_size - train_size

train_dataset, val_dataset, _ = data_utils.random_split(dataset=full_dataset, lengths=[train_size, valdata_size, remaining_size],
                                                     generator=torch.Generator().manual_seed(0))
train_loader = data_utils.DataLoader(train_dataset, batch_size=BatchSize, shuffle=True, drop_last=True, num_workers=6)
val_loader = data_utils.DataLoader(val_dataset, batch_size=BatchSize, shuffle=True, drop_last=True, num_workers=6)

print(f"总数据集大小: {len(full_dataset)}")
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")

print('           START TRAINING                  ')

print('Original data dimention:%s' % str(DataDim))
print('Downsampled data dimention:%s ' % str(data_dsp_dim))
print('Original label dimention:%s' % str(ModelDim))
print('Downsampled label dimention:%s' % str(label_dsp_dim))
print('Training size:%d' % int(TrainSize))
print('Training batch size:%d' % int(BatchSize))
print('Number of epochs:%d' % int(Epochs))
print('Learning rate:%.5f' % float(LearnRate))

# 保存 scaler 对象到文件，供测试阶段使用
joblib.dump(scaler, main_dir+'scaler.pkl')
print("Scaler 已保存到 'scaler.pkl'")

# Initialization
train_loss = []
val_loss = []
# step   = np.int(TrainSize/BatchSize)
step = np.int(train_set.shape[0] // BatchSize)
start = time.time()

best_psnr = 0
best_epoch = 0
best_iter = 0


for epoch in range(Epochs):
    # epoch_loss = 0.0
    train_epoch_loss = 0.0
    val_epoch_loss = 0.0
    since = time.time()

    #### Trainning ####
    # Set Net with train condition
    net.train() # tqdm
    for i, (images, labels) in enumerate(train_loader):
        iteration = epoch * step + i + 1

        # Reshape data size
        # 假设输入数据大小为 [batch_size, channels, depth, height, width]
        images = images.view(BatchSize, 1, Inchannels, Patch_size, Patch_size).contiguous()
        labels = labels.view(BatchSize, 1, Nclasses, Patch_size, Patch_size).contiguous()
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward prediction
        # outputs = net(images,[1,1])
        outputs = net(images)

        # Calculate the MSE
        # loss = F.mse_loss(outputs, images - labels, reduction='sum') / (BatchSize)
        # loss = F.mse_loss(outputs, labels, reduction='sum') / (BatchSize)
        #criterion = MS_SSIM_L1_LOSS()
        #loss=criterion(outputs,labels)

        # Calculate the MSE and GDL
        GDL_loss = MSE_and_3D_GDL()
        loss = GDL_loss(outputs,images - labels, 1, 1)
        # loss = GDL_loss(outputs,labels, 1, 1)

        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while training')

        train_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        # Print loss
        if iteration % DisplayStep == 0:
            print('Epoch: {}/{}, Iteration: {}/{} --- Training Loss:{:.6f}'.format(epoch + 1, \
                                                                                   Epochs, iteration, \
                                                                                   step * Epochs, loss.item()))
    train_epoch_loss = train_epoch_loss / (i + 1)

    #### Evaluation ####
    # use validaData to validatation net
    net.eval()
    snr_val_total = []
    for batch_idx2, (val_images, val_labels) in enumerate(val_loader):
        iteration = epoch * step + batch_idx2 + 1

        # Reshape data size
        # 假设输入数据大小为 [batch_size, channels, depth, height, width]
        val_images = val_images.view(BatchSize, 1, Inchannels, Patch_size, Patch_size).contiguous()
        val_labels = val_labels.view(BatchSize, 1, Nclasses, Patch_size, Patch_size).contiguous()
        val_images = val_images.to(device)
        val_labels = val_labels.to(device)

        with torch.no_grad():
            outputs = net(val_images)

            # snr_val_total.append(torchSNR(outputs, val_images - val_labels))
            snr_val_total.append(torchSNR(val_images- outputs, val_labels))
            # snr_val_total.append(torchSNR(outputs, val_labels))

            # # Calculate the MSE
            # loss = F.mse_loss(outputs, val_images - val_labels, reduction='sum') / (BatchSize)

            # Calculate the MSE and GDL
            GDL_loss = MSE_and_3D_GDL()
            loss = GDL_loss(outputs, val_images - val_labels, 1, 1)

            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while validation')

            val_epoch_loss += loss.item()
            # Print loss
            if iteration % DisplayStep == 0:
                print('Epoch: {}/{}, Iteration: {}/{} --- Validation Loss:{:.6f}'.format(epoch + 1, \
                                                                                         Epochs, iteration, \
                                                                                         step * Epochs,
                                                                                         loss.item()))
    val_epoch_loss = val_epoch_loss / (batch_idx2 + 1)
    scheduler.step()

    #save best model
    snr_val_total = torch.stack(snr_val_total).mean().item()
    if snr_val_total > best_psnr:
        best_psnr = snr_val_total
        best_epoch = epoch
        best_iter = i
        torch.save(net.state_dict(), models_dir + modelname + 'model_best.pkl')
    print("[epoch %d it %d PSNR: %.4f --- best_epoch %d best_iter %d Best_PSNR %.4f]" % (epoch, i, snr_val_total, best_epoch, best_iter, best_psnr))

    # Print loss and consuming time every epoch
    if (epoch + 1) % 1 == 0:
        print('Epoch: {:d} finished ! Training Loss: {:.5f}'.format(epoch + 1, train_epoch_loss))
        train_loss = np.append(train_loss, train_epoch_loss)
        print('Epoch: {:d} finished ! Validating Loss: {:.5f}'.format(epoch + 1, val_epoch_loss))
        val_loss = np.append(val_loss, val_epoch_loss)
        time_elapsed = time.time() - since
        print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # Save net parameters every 10 epochs
    if (epoch + 1) % SaveEpoch == 0:
        torch.save(net.state_dict(), models_dir + modelname + '_epoch' + str(epoch + 1) + '.pkl')
        print('Trained model saved: %d percent completed' % int((epoch + 1) * 100 / Epochs))

    # Record the consuming time
time_elapsed = time.time() - start
print('Training complete in {:.0f}m  {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# Save the loss
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 17,
         }
font3 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 21,
         }
SaveTrainResults(train_loss=train_loss, val_loss=val_loss, SavePath=results_dir, font2=font2, font3=font3)
