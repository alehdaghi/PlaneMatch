import os
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataset_planematch2 import *
from network_planematch2 import *
import util

config = util.get_args()
transformed_dataset = PlanarPatchDataset(csv_file=config.test_csv_path, root_dir=config.test_root_dir,transform=transforms.Compose([ToTensor()]))
dataloader = DataLoader(transformed_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
triplet_loss = nn.TripletMarginLoss(margin=1.0)

model = torch.load('./model.pkl')
model.eval()
#model.cuda(config.gpu)

if not os.path.exists(config.feature_path):
    os.makedirs(config.feature_path)

count = 0
print('Start testing ......')
for i_batch, sample_batched in enumerate(dataloader):
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

    feature1 = model(x1, x2, x3)
    feature2 = model(x4, x5, x6)
    feature3 = model(x7, x8, x9)

    loss = triplet_loss(feature1, feature2, feature3)
    loss.data = (loss.data / 1) ** (config.focal_loss_lambda)
    focal_loss = loss.data
    dist1 = np.linalg.norm(feature1.data.cpu().numpy() - feature2.data.cpu().numpy())
    dist2 = np.linalg.norm(feature1.data.cpu().numpy() - feature3.data.cpu().numpy())
    print('feature distanceP: ',dist1, ' distanceN: ', dist2)
    print('[%d] focal_loss: %.16f' % (i_batch + 1, focal_loss))

    # if count == 0:
    #     featureAll1 = feature1.data.cpu().numpy()
    #     featureAll2 = feature2.data.cpu().numpy()
    #     featureAll3 = feature3.data.cpu().numpy()
    # else:
    #     featureAll1 = np.vstack((featureAll1,feature1.data.cpu().numpy()))
    #     featureAll2 = np.vstack((featureAll2,feature2.data.cpu().numpy()))
    #     featureAll3 = np.vstack((featureAll3, feature2.data.cpu().numpy()))
    # for i in range(0,feature1.data.cpu().numpy().shape[0]):
    #     dist1 = np.linalg.norm(feature1.data.cpu().numpy()[i] - feature2.data.cpu().numpy()[i])
    #     dist2 = np.linalg.norm(feature1.data.cpu().numpy()[i] - feature3.data.cpu().numpy()[i])
    #     print('feature distanceP: ',dist1, ' distanceN: ', dist2)
    count = count + 1
    
#print('Writing feature to file...')
#np.savetxt(config.feature_path+"/feature1.txt",featureAll1, fmt="%f", delimiter="  ")
#np.savetxt(config.feature_path+"/feature2.txt",featureAll2, fmt="%f", delimiter="  ")
#np.savetxt(config.feature_path+"/feature3.txt",featureAll3, fmt="%f", delimiter="  ")
print('Finished...')
