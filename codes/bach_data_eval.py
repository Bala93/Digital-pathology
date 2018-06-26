import os



import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self,index):
        path,target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img,target,path

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/media/htic/NewVolume1/murali/mitosis/bach18/scale_16_train_val'
image_datasets = {x: MyImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
use_gpu = torch.cuda.is_available()


if __name__ == "__main__":

    running_corrects = 0
    model_path = '/media/htic/NewVolume1/murali/mitosis/bach18/model/bach18.pt'

    # model= models.resnet101()
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)
    model = torch.load(model_path)
    
    if use_gpu:
        model = model.cuda()

    with open('/media/htic/NewVolume1/murali/mitosis/bach18/model/path.txt','w') as f:
        for data in tqdm(dataloaders['val']):

            inputs,labels,path = data
            f.write(path[0] + '\n')

            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs,labels = Variable(inputs),Variable(labels)

            outputs = model(inputs)
            _,preds = torch.max(outputs.data,1)

            running_corrects += torch.sum(preds == labels.data)

        print dataset_sizes['val']
        print running_corrects
        epoch_acc = float(running_corrects)/dataset_sizes['val']
        print (epoch_acc)
    # print ("Accuracy:{.4f}".format(epoch_acc))