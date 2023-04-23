import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision.io import read_image
import torchvision.transforms as transforms
import random
from PIL import Image

def get_one_cate_train_val_test_files(file_location, cate_num, val_test_ratio = 0.3, seed = 1):
    '''
    Obtain train and test file path of one category and its corresponding label
    '''
    file_names = os.listdir(file_location)
    n = len(file_names)
    random.Random(seed).shuffle(file_names)
    split_index = int(n * val_test_ratio)
    split_index_half = int(split_index//2)
    val = [os.path.join(file_location, x) for x in file_names[:split_index_half]]
    testing = [os.path.join(file_location, x) for x in file_names[split_index_half:split_index]]
    training = [os.path.join(file_location, x) for x in file_names[split_index:]]
    labels_val = torch.Tensor([cate_num]).repeat(len(val))
    labels_testing = torch.Tensor([cate_num]).repeat(len(testing))
    labels_training = torch.Tensor([cate_num]).repeat(len(training))
    return training, labels_training, val, labels_val, testing, labels_testing

def get_train_test_names_labels(file_location, val_test_ratio = 0.3, seed = 1):
    '''
    Obtain train and test file path for all corresponding categories and their corresponding label
    '''
    assert 'coronal' in os.listdir(file_location), 'folder named coronal must be the file_location directory'
    training_h, labels_training_h, val_h, labels_val_h, testing_h, labels_testing_h = get_one_cate_train_val_test_files(os.path.join(file_location, 'horizontal'), 
                                                                                               cate_num = 0, val_test_ratio = val_test_ratio, seed = seed)
    training_c, labels_training_c, val_c, labels_val_c,  testing_c, labels_testing_c = get_one_cate_train_val_test_files(os.path.join(file_location, 'coronal'), 
                                                                                               cate_num = 1, val_test_ratio = val_test_ratio, seed = seed)
    training_s, labels_training_s, val_s, labels_val_s, testing_s, labels_testing_s = get_one_cate_train_val_test_files(os.path.join(file_location, 'sagittal'), 
                                                                                               cate_num = 2, val_test_ratio = val_test_ratio, seed = seed)
    train_val_test_num_dict = {
        'Horizontal train':len(training_h), 'Coronal train': len(training_c), 'Sagittal train': len(training_s),
        'Horizontal val':len(val_h), 'Coronal val': len(val_c), 'Sagittal val': len(val_s),
        'Horizontal testing':len(testing_h), 'Coronal testing': len(testing_c), 'Sagittal testing': len(testing_s),        
               }
    train_file_paths = training_h + training_c + training_s
    train_labels = torch.cat((labels_training_h, labels_training_c, labels_training_s))
    
    val_file_paths = val_h + val_c + val_s
    val_labels = torch.cat((labels_val_h, labels_val_c, labels_val_s))
    
    test_file_paths = testing_h + testing_c + testing_s
    test_labels = torch.cat((labels_testing_h, labels_testing_c, labels_testing_s))
    
    return train_file_paths, train_labels, val_file_paths, val_labels, test_file_paths, test_labels, train_val_test_num_dict

class brain_image_datset(Dataset):
    def __init__(self, file_paths, labels, transform = None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image =  Image.open(self.file_paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image.double(), label.long(), self.file_paths[idx]

class numpy_datset(Dataset):
    def __init__(self, images, transform = None):
        self.images = images
        self.transform = transform
    def __len__(self):
        return self.images.shape[0]
    def __getitem__(self, idx):
        image = self.images[idx, :, :, :]
        if self.transform is not None:
            image = self.transform(image)
        return image.double()
    
    
def cal_train_mean_sd(file_paths, labels):
    '''
    Calculate the full data mean and std for the training set 
    '''
    transform = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Resize([512, 512])
        ]
    )
    train_dataloader = DataLoader(brain_image_datset(file_paths = file_paths,labels = labels, transform = transform), 
                                  batch_size = 128, shuffle = True)
    mean = 0.
    meansq = 0.
    for images, _, _ in train_dataloader:
        mean += images.mean(axis = (0,2,3))
        meansq += (images**2).mean(axis = (0,2,3))
    mean /= len(train_dataloader)
    meansq /= len(train_dataloader)
    #import pdb; pdb.set_trace()
    return mean.tolist(), torch.sqrt(meansq-mean**2).tolist()