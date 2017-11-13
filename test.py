from dataset import *
from model import Net

import argparse
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.tensor
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('dataroot', help='path to test dataset ')
# parser.add_argument('dataroot', default='data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')

parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--useBN', action='store_true', help='enalbes batch normalization')

args = parser.parse_args()
print(args)

dataset = kaggle2016nerve(args.dataroot,train=False)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                           num_workers=args.workers, shuffle=False)

model = Net(args.useBN)

if args.cuda:
  model.cuda()
  cudnn.benchmark = True

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))

        if args.cuda == False:
            checkpoint = torch.load(args.resume, map_location={'cuda:0': 'cpu'})
        else:
            checkpoint = torch.load(args.resume)

        args.start_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {}, loss {})"
              .format(checkpoint['epoch'], checkpoint['loss']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        assert False
else:
    print("Please input the check point files")


def showImg(img, binary=True, fName=''):
  """
  show image from given numpy image
  """
  img = img[0,0,:,:]

  if binary:
    img = img > 0.5

  img = Image.fromarray(np.uint8(img*255), mode='L')

  if fName:
    img.save('assets/'+fName+'.png')
  else:
    img.show()

model.eval()
train_loader.batch_size=args.batchSize

for i, (x,img_name) in enumerate(train_loader):
  #print(img_name[0])
  file_name = img_name[0]
  print("inferece: %s "%file_name)
  y_pred = model(Variable(x.cuda()))
  showImg(x.numpy(), binary=False, fName=file_name)
  showImg(y_pred.cpu().data.numpy(), binary=True, fName=file_name+'_pred')
  #showImg(y.numpy(), fName='gt_'+str(i))
