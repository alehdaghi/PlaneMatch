import os
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from time import gmtime, strftime, clock
from dataset_planematch2 import *
from network_planematch2 import *
import util


config = util.get_args()
if not config.disable_cuda and torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')
transformed_dataset = PlanarPatchDataset(csv_file=config.train_csv_path, root_dir=config.train_root_dir,transform=transforms.Compose([ToTensor()]))
dataloader = DataLoader(transformed_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

model = ResNetMI(Bottleneck, [3, 4, 6, 3])
#model = torch.load(os.path.join(config.save_path, 'snapshots_2019-09-17_05-44-13/model120.pkl'), map_location=torch.device('cpu'))
#model.eval()
#model.cuda(config.gpu)

triplet_loss = nn.TripletMarginLoss(margin=1.0)
learning_rate = config.lr
iteration = 0

if config.save_snapshot:
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    snapshot_folder = os.path.join(config.save_path, 'snapshots_'+strftime("%Y-%m-%d_%H-%M-%S",gmtime()))
    if not os.path.exists(snapshot_folder):
        os.makedirs(snapshot_folder)

print('Start training ......')
enumerate1 = enumerate(dataloader)
for epoch in range(config.epochs):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch)
        sample = transformed_dataset[i_batch]
        x1 = sample_batched['rgb_anchor_image'].float()
        x2 = sample_batched['normal_anchor_image'].float()
        x3 = sample_batched['mask_anchor_image'].float()
        x4 = sample_batched['rgb_positive_image'].float()
        x5 = sample_batched['normal_positive_image'].float()
        x6 = sample_batched['mask_positive_image'].float()
        x7 = sample_batched['rgb_negative_image'].float()
        x8 = sample_batched['normal_negative_image'].float()
        x9 = sample_batched['mask_negative_image'].float()



        # x1 = Variable(x1.cuda(config.gpu), requires_grad=True)
        # x2 = Variable(x2.cuda(config.gpu), requires_grad=True)
        # x3 = Variable(x3.cuda(config.gpu), requires_grad=True)
        # x4 = Variable(x4.cuda(config.gpu), requires_grad=True)
        # x5 = Variable(x5.cuda(config.gpu), requires_grad=True)
        # x6 = Variable(x6.cuda(config.gpu), requires_grad=True)
        # x7 = Variable(x7.cuda(config.gpu), requires_grad=True)
        # x8 = Variable(x8.cuda(config.gpu), requires_grad=True)
        # x9 = Variable(x9.cuda(config.gpu), requires_grad=True)
        # x10 = Variable(x10.cuda(config.gpu), requires_grad=True)
        # x11 = Variable(x11.cuda(config.gpu), requires_grad=True)
        # x12 = Variable(x12.cuda(config.gpu), requires_grad=True)
        # x13 = Variable(x13.cuda(config.gpu), requires_grad=True)
        # x14 = Variable(x14.cuda(config.gpu), requires_grad=True)
        # x15 = Variable(x15.cuda(config.gpu), requires_grad=True)
        # x16 = Variable(x16.cuda(config.gpu), requires_grad=True)
        # x17 = Variable(x17.cuda(config.gpu), requires_grad=True)
        # x18 = Variable(x18.cuda(config.gpu), requires_grad=True)
        # x19 = Variable(x19.cuda(config.gpu), requires_grad=True)
        # x20 = Variable(x20.cuda(config.gpu), requires_grad=True)
        # x21 = Variable(x21.cuda(config.gpu), requires_grad=True)
        # x22 = Variable(x22.cuda(config.gpu), requires_grad=True)
        # x23 = Variable(x23.cuda(config.gpu), requires_grad=True)
        # x24 = Variable(x24.cuda(config.gpu), requires_grad=True)

        feature1 = model(x1, x2, x3)
        feature2 = model(x4, x5, x6)
        feature3 = model(x7, x8, x9)
        #
        loss = triplet_loss(feature1, feature2, feature3)
        loss.data = (loss.data/1)**(config.focal_loss_lambda)
        focal_loss = loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('[%d, %d]   lr: %.9f   focal_loss: %.16f' %(epoch + 1, i_batch + 1, learning_rate, focal_loss))

        if config.save_snapshot and iteration % config.save_snapshot_every == 0 :
            print('Saving snapshots of the models ...... ')
            torch.save(model, snapshot_folder+'/model'+ str(iteration) + '.pkl')
        if iteration % config.lr_decay_every == config.lr_decay_every - 1:
            learning_rate = learning_rate * config.lr_decay_by
        iteration = iteration + 1

print('Saving snapshots of the models ...... ')
torch.save(model, snapshot_folder+'/model'+ str(iteration) + '.pkl')
print('Finished..')
