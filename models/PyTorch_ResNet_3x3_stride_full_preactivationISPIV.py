import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
#import torch.utils.data.DataLoader
import numpy as np
import time
import datetime


class H5Dataset(data.Dataset):

    """def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        self.h5_file = h5py.File(file_path, 'r')
        self.data = self.h5_file['ComImages']
        self.target = self.h5_file['AllGenDetails']

    def __getitem__(self, index):            
        return (torch.from_numpy(self.data[index,:,:,:]).float(),
                torch.from_numpy(self.target[index,:,:,:]).float())

    def __len__(self):
        return self.data.shape[0]"""

    def __init__(self, hdf5file, imgs_key='ComImages', labels_key='AllGenDetails',
                 transform=None):
        '''
        :argument
        :param hdf5file: the hdf5 file including the images and the label.
        :param transform (callable, optional): Optional transform to be
        applied on a sample
        '''
        self.db = h5py.File(hdf5file, 'r')  # store the images and the labels
        keys = list(self.db.keys())
        if imgs_key not in keys:
            raise (' the ims_key should not be {}, should be one of {}'
                   .format(imgs_key, keys))
        if labels_key not in keys:
            raise (' the labels_key should not be {}, should be one of {}'
                   .format(labels_key, keys))
        self.imgs_key = imgs_key
        self.labels_key = labels_key
        #self.transform = transform

    def __len__(self):
        return len(self.db[self.labels_key])

    def __getitem__(self, idx):
        image = np.transpose(self.db[self.imgs_key][idx], (2 , 1, 0)) #NHWC -> NCHW
        label = self.db[self.labels_key][idx]
        sample = {self.imgs_key: image, self.labels_key: label}
        #if self.transform:
        #    sample = self.transform(sample)
        return sample




class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, prev_layer_depth, expansion=4, stride_3x3=1,  padding_3x3=1, conv_identity=False, stride_conv_identity=1, activation_normalization=True):
        super(Bottleneck, self).__init__()
        self.activation_normalization = activation_normalization
        self.outplanes = planes*expansion
        self.conv_identity = conv_identity
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1)
        self.conv1_bn = nn.BatchNorm2d(prev_layer_depth)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride_3x3, padding=padding_3x3)
        self.conv2_bn = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=expansion*planes, kernel_size=1)
        self.conv3_bn = nn.BatchNorm2d(planes)
        if conv_identity:
            self.conv_identity_layer = nn.Conv2d(inplanes, planes*expansion, kernel_size=1, stride=stride_conv_identity)

    def forward(self, x):
        identity = x

        if self.activation_normalization:
            out = self.conv1(F.relu(self.conv1_bn(x)))
            out = self.conv2(F.relu(self.conv2_bn(out)))
            out = self.conv3(F.relu(self.conv3_bn(out)))

        else:
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)

        if self.conv_identity:
            identity = self.conv_identity_layer(x)

        out += identity

        return out

class Net(nn.Module):

   def __init__(self, batch_size):
      super(Net, self).__init__()

      self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
      self.conv1_bn = nn.BatchNorm2d(16)

      self.stage_1a = self._make_layer(inplanes=16, planes=16, prev_layer_depth=16, blocks=1, conv_identity=True)
      self.stage_1b = self._make_layer(inplanes=64, planes=16, prev_layer_depth=64, blocks=2)
      self.stage_2a = self._make_layer(inplanes=64, planes=32, prev_layer_depth=64, blocks=1, stride_3x3=2, conv_identity=True,  stride_conv_identitiy=2)
      self.stage_2b = self._make_layer(inplanes=128, planes=32, prev_layer_depth=128, blocks=3)
      self.stage_3a = self._make_layer(inplanes=128, planes=64, prev_layer_depth=128, blocks=1, stride_3x3=2, conv_identity=True, stride_conv_identitiy=2)
      self.stage_3b = self._make_layer(inplanes=256, planes=64, prev_layer_depth=256, blocks=5)
      self.stage_4a = self._make_layer(inplanes=256, planes=128, prev_layer_depth=256, blocks=1, stride_3x3=2, conv_identity=True, stride_conv_identitiy=2)
      self.stage_4b = self._make_layer(inplanes=512, planes=128, prev_layer_depth=512, blocks=2)

      self.ANN5 = nn.Linear(8192, 4096)
      self.ANN5_bn = nn.BatchNorm1d(4096)
      self.ANN6 = nn.Linear(4096, 2048)
      self.ANN6_bn = nn.BatchNorm1d(2048)
      self.ANN7 = nn.Linear(2048, 1024)
      self.ANN7_bn = nn.BatchNorm1d(1024)
      self.ANN8 = nn.Linear(1024, 2)

   def forward(self, input):
       output = F.relu(self.conv1_bn(self.conv1(input)))
       output = self.stage_1a(output)
       output = self.stage_1b(output)
       output = self.stage_2a(output)
       output = self.stage_2b(output)
       output = self.stage_3a(output)
       output = self.stage_3b(output)
       output = self.stage_4a(output)
       output = self.stage_4b(output)

       output = F.leaky_relu(self.ANN5_bn(self.ANN5(output.reshape([input.shape[0], 512*4*4]))), negative_slope=0.1)
       output = F.leaky_relu(self.ANN6_bn(self.ANN6(output)), negative_slope=0.1)
       output = F.leaky_relu(self.ANN7_bn(self.ANN7(output)), negative_slope=0.1)
       output = self.ANN8(output)

       return output

   def _make_layer(self, inplanes, planes, blocks, prev_layer_depth, expansion=4, stride_3x3=1, padding_3x3=1, conv_identity=True, stride_conv_identitiy=1, activation_normalization=True):
      layers = []
      for _ in range(blocks):
         layers.append(Bottleneck(inplanes, planes, prev_layer_depth=prev_layer_depth, stride_3x3=stride_3x3, padding_3x3=padding_3x3, conv_identity=conv_identity, stride_conv_identity=stride_conv_identitiy, activation_normalization=activation_normalization))

      return nn.Sequential(*layers)


name = 'ISPIV_ResNet_3x3_full_preactivation_IPI_D4'
training_begin = time.strftime("%d-%m-%y_%H:%M:%S")
batchSize = 1000

training_set = H5Dataset('/home/bahr/cdm/data/ISPIV_dataset/Batch_Training-Dataset_2Labels_S12_SynthImg_Alex.hdf5')
training_generator = torch.utils.data.DataLoader(training_set, batch_size=batchSize, shuffle=True, num_workers=1)
validation_set = H5Dataset('/home/bahr/cdm/data/ISPIV_dataset/Batch_Validation-Dataset_2Labels_S12_SynthImg_Alex.hdf5')
validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=batchSize, shuffle=True, num_workers=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)
#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
neural_net = Net(batchSize).to(device)

initial_lr = 0.0025
optimizer = torch.optim.Adam(neural_net.parameters(), lr=initial_lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=10, min_lr=1e-8)

training_loss_function = loss = nn.MSELoss()
validation_loss_function = nn.L1Loss()
print(neural_net)
max_epochs = 200
start_epoch = 0

recover = False
if recover:
    checkpoint = torch.load("/home/giganto/Nets/ISPIV/PyTorch/Checkpoints/ISPIV_ResNet_3x3_full_preactivation_IPI_D4_20-06-19_10:18:02.tar")
    neural_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']+1
    print("recovering at epoch: ", start_epoch)



print("starting training")

log_file = open('/home/bahr/cdm/resnet/'+name+'_'+training_begin+'.txt', "w")

# Loop over epochs
for epoch in range(start_epoch, max_epochs):
    # Training
    print(epoch)
    last_training_loss = 0.0
    las_validation_loss = 0.0
    neural_net.train()
    t0 = time.time()
    for i_batch, sample_batched in enumerate(training_generator):
        #for i, data in enumerate(dataloader, 0):
        # Transfer to GPU
        local_batch, local_labels = sample_batched['ComImages'].float().to(device), sample_batched['AllGenDetails'].float().to(device)

        # Model computations
        training_pred = neural_net.forward(local_batch)
        training_loss = training_loss_function(training_pred, local_labels)
        last_training_loss = training_loss
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()
        if(i_batch%50 == 0):
            print("[",epoch,"/",max_epochs-1,"] [", i_batch, "/", len(training_generator)-1, "] elapsed time", str(datetime.timedelta(seconds=time.time()-t0)), "training_loss: ", training_loss.item(), "validation_loss = --", " learning rate = ", optimizer.__getattribute__('param_groups')[0]['lr'],end="\r")



    # Validation
    with torch.set_grad_enabled(False):
        neural_net.eval()
        for i_batch, sample_batched in enumerate(validation_generator):
            # Transfer to GPU
            local_batch, local_labels = sample_batched['ComImages'].float().to(device), sample_batched['AllGenDetails'].float().to(device)

            # Model computations
            validation_pred = neural_net.forward(local_batch)
            validation_loss = validation_loss_function(validation_pred, local_labels)
            last_validation_loss = validation_loss

            if (i_batch % 50 == 0):
                print("[", epoch, "/", max_epochs - 1, "] [", i_batch, "/", len(validation_generator)-1, "] elapsed time: ", str(datetime.timedelta(seconds=time.time() - t0)), "training_loss: ", last_training_loss.item(), " validation_loss = ", validation_loss.item(), " learning rate = ", optimizer.__getattribute__('param_groups')[0]['lr'], end="\r")

    scheduler.step(validation_loss)
    print('\n')
    if(epoch%5 == 0):
        torch.save({
            'epoch': epoch,
            'model_state_dict': neural_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, '/home/bahr/cdm/resnet/'+name+'_'+training_begin+'.pth')
    log_file.write("epoch: "+str(epoch)+" training_loss: "+str(last_training_loss.item())+" validation loss: "+str(last_validation_loss.item())+"\n")
    log_file.flush()
