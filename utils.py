from __future__ import print_function
import os.path
import torch.utils.data as data
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image
import torchvision.models as models
import keras.backend as K

class Feline(data.Dataset):    
    def __init__(self, imagedic, Deblur=False, Blured=False,
                 transform=None, target_transform=None, blur = None):
        self.transform = transform
        self.target_transform = target_transform

        # now load the numpy arrays
        entry = imagedic
        if Blured:
            self.data = entry['blur_data']
        elif Deblur:
            self.data = entry['deblur_data']
        else:
            self.data = entry['data']
        self.labels = entry['label']
        self.data = self.data.reshape((-1, 64, 64, 3))
                
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    

def sampleAccuracy(dataloader, net):
    correct = 0
    total = 0
    for data in dataloader:
        inputs, labels = data       
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = net(inputs)
        
        _, predicted = torch.max(outputs.data, 1)        
        total += labels.size(0)
        labels = labels.view(labels.size()[0])
        correct += (predicted == labels.data).sum()
    return correct.cpu().data.numpy() / total

def sampleLoss(dataloader, net):
    running_loss = 0
    for i, data in enumerate(dataloader, 0):        
        inputs, labels = data       
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())       
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)       
        running_loss += loss.cpu().data[0]       
    res = running_loss/len(dataloader)
    return res

def VGGpredict(dataloader, net):
    predict_label = []
    for data in dataloader:
        inputs, labels = data       
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predict_label.append(predicted)
    return predict_label

def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))