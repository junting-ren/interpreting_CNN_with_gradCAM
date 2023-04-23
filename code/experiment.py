import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import datetime
from constants import ROOT_STATS_DIR
from data_loader import *
from file_utils import *
from model_constructor import get_model

import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Experiment(object):
    """
    Experiment object class for loading data, training, validation, and testing
    
    Methods
    --------
    _load_experiment: load the experiment and train from last check point
    run: start the training and validation process
    _train: train the model
    _val:validate the model
    test: evaludate the model on future data where the outcome could be unknown
    cal_attend_rate: calculate the attendance rate from predicted worked hours
    _save_model: save the LSTM model 
    _record_stats: record the loss and metrics
    plot_stats: plot the loss and metrics
    """
    def __init__(self, config_data):
        """Initialize the parameters from config_data
        
        Parameters
        ------------
        config_data: a dictionary contains all the configuration parameters
        
        Returns
        ---------
        None
        """
        # config_data = read_file_in_dir('./', name + '.json')
        # if config_data is None:
        #     raise Exception("Configuration file doesn't exist: ", name)
        pd.options.mode.chained_assignment = None  # default='warn'
        if torch.cuda.is_available():
            self.device = config_data['device']
        else:
            self.device = 'cpu'
        self._name = config_data['experiment_name']
        self.batchsize = config_data['batch_size']
        self.seed = config_data['seed']
        self._experiment_dir = os.path.join(ROOT_STATS_DIR, self._name)
        # Setup Experiment
        self.scheduler_decay_rate = config_data['decay_rate']
        self._weight_decay = config_data['l2_p']
        self._epochs = config_data['num_epochs']
        self._lr = config_data['lr']
        self._current_epoch = 0
        self.patience = config_data['patience']
        self._training_losses = []
        self._val_losses = []
        self._val_metric = []
        self._best_model = None  # Save your best model in this field and use this in test method.
        # Criterion and Optimizers 
        self._criterion = torch.nn.CrossEntropyLoss()
        # Init Model
        self._model = get_model(config_data, self.device)  
        self._optimizer = optim.Adam(self._model.parameters(), lr = self._lr, weight_decay= self._weight_decay)
        self._scheduler = ExponentialLR(self._optimizer, gamma=self.scheduler_decay_rate)
        self._init_model()
        #Load Experiment Data if available
        self._load_experiment() 
        # write the current parameters dict
        file = open(os.path.join(self._experiment_dir,"config_file.txt"),"w")
        for key, value in config_data.items():
            file.write('%s:%s\n' % (key, value))
        file.close()
        self.eval_flag = config_data['eval_flag']
        # Load Datasets
        if not self.eval_flag:
            (train_file_paths, train_labels, val_file_paths, 
             val_labels, test_file_paths, test_labels,
            self.train_val_test_num_dict
            ) = get_train_test_names_labels(config_data['data_path'], 
                                            val_test_ratio = config_data['val_test_ratio'], 
                                            seed = self.seed)
            self.train_file_paths = train_file_paths
            self.val_file_paths = val_file_paths
            self.test_file_paths = test_file_paths
            self.mean_train, self.sd_train = cal_train_mean_sd(train_file_paths, train_labels)
            write_to_file_in_dir(self._experiment_dir, 'mean_train.txt', self.mean_train)
            write_to_file_in_dir(self._experiment_dir, 'sd_train.txt', self.sd_train)
            transform_train = transforms.Compose(
                [
                transforms.ToTensor(),
                transforms.Resize([512, 512]),
                transforms.Normalize(mean = self.mean_train, std = self.sd_train), 
                transforms.RandomApply([transforms.RandomResizedCrop(size = [512, 512], scale=(0.5, 1.0)),
                                        transforms.RandomRotation(degrees = [30, 180])
                                       ], p =0.5)
                ]
            )
            transform_val_test = transforms.Compose(
                [
                transforms.ToTensor(),
                transforms.Resize([512, 512]),
                transforms.Normalize(mean = self.mean_train, std = self.sd_train)
                ]
            )
            self.train_dataloader = DataLoader(brain_image_datset(file_paths = train_file_paths,labels = train_labels,
                                                                  transform = transform_train ), batch_size = 32, shuffle = True)
            self.val_dataloader = DataLoader(brain_image_datset(file_paths = val_file_paths,labels = val_labels,
                                                                transform = transform_val_test ), batch_size = 32, shuffle = True)
            self.test_dataloader = DataLoader(brain_image_datset(file_paths = test_file_paths,labels = test_labels,
                                                                 transform = transform_val_test), batch_size = 32, shuffle = True)

            self.train_size = len(self.train_dataloader)
    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def _load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)
        if os.path.exists(self._experiment_dir) and os.path.exists(os.path.join(self._experiment_dir, 'training_losses.txt')):
            self._training_losses = read_file_in_dir(self._experiment_dir, 'training_losses.txt')
            self._val_losses = read_file_in_dir(self._experiment_dir, 'val_losses.txt')
            self._val_metric = read_file_in_dir(self._experiment_dir, 'val_metric.txt')
            self._current_epoch = len(self._training_losses)
            self.mean_train = read_file_in_dir(self._experiment_dir, 'mean_train.txt')
            self.sd_train = read_file_in_dir(self._experiment_dir, 'sd_train.txt')
            state_dict = torch.load(os.path.join(self._experiment_dir, 'latest_model.pt'))
            self._model.load_state_dict(state_dict['model'])
            self._optimizer.load_state_dict(state_dict['optimizer'])
        else:
            os.makedirs(self._experiment_dir, exist_ok  = True)

    def _init_model(self):
        self._model = self._model.to(self.device).double()
        self._criterion = self._criterion.to(self.device)

    # Main method to run experiment. Should be self-explanatory.
    def run(self):
        #import pdb; pdb.set_trace()
        start_epoch = self._current_epoch
        min_val_loss = float('inf')
        patient = 0
        for epoch in range(start_epoch, self._epochs):  # loop over the dataset multiple times
            start_time = datetime.datetime.now()
            train_loss = self._train()
            val_loss, val_metric = self._val()
            if(val_loss < min_val_loss):
                print("Saving the best model after {} epochs".format(epoch))
                min_val_loss = val_loss
                self._save_model(name = 'best_model.pt')
                patient = 0
            else:
                patient += 1
            self._current_epoch = epoch
            self._record_stats(train_loss, val_loss, val_metric)
            self._log_epoch_stats(start_time)
            self._save_model()
            # print('Validation loss is '+ str(val_loss) + ' and mae worked hour is '+ str(val_mae_worked_hour)+ ' and termination accuracy is ' + str(val_acc_term) + ' at epoch ' + str(epoch))
            # print('Training loss is '+ str(train_loss) + ' at epoch ' + str(epoch))
            # print('---------------------------------------------------------------')
            if patient > self.patience:
                break        

    def _train(self):
        self._model.train()
        training_loss = 0
        for i, (images, labels, file_paths) in enumerate(self.train_dataloader):
            self._model.zero_grad()
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self._model(images)
            loss = self._criterion(outputs, labels)
            training_loss += loss.item()
            loss.backward()
            self._optimizer.step()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
        self._scheduler.step()
        return training_loss/len(self.train_dataloader)        

    def _val(self):
        self._model.eval()
        val_loss = 0
        val_metric = [] 
        for i, (images, labels, file_paths)  in enumerate(self.val_dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self._model(images)
            #import pdb; pdb.set_trace()
            loss = self._criterion(outputs, labels)
            #import pdb; pdb.set_trace()
            val_m = np.mean((torch.argmax(outputs,dim = 1)==labels).detach().cpu().numpy())
            val_loss += loss.item()
            val_metric.append(val_m)

        return val_loss/len(self.val_dataloader), np.mean(val_metric)

    def test(self):
        state_dict = torch.load(os.path.join(self._experiment_dir, 'best_model.pt'))
        self._model.load_state_dict(state_dict['model'])
        self._model.eval()
        result_l = [] 
        total_n = len(self.test_dataloader)
        pred_labels = []
        true_labels = []
        file_paths_l = []
        cross_entropy_l = []
        for i, (images, labels, file_paths) in enumerate(self.test_dataloader):                
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self._model(images)
            cross_entropy_l.extend(F.cross_entropy(outputs, labels, reduction = 'none').detach().cpu().numpy().tolist())
            true_labels.extend(labels.data.cpu().numpy())
            pred_labels.extend(torch.argmax(outputs,dim = 1).data.cpu().numpy())
            file_paths_l.extend(file_paths)
            result_l.append(np.mean((torch.argmax(outputs,dim = 1)==labels).detach().cpu().numpy()))
        return np.mean(result_l),pred_labels, true_labels, cross_entropy_l, file_paths_l

    
    
    def _save_model(self, name = 'latest_model.pt'):
        root_model_path = os.path.join(self._experiment_dir, name)
        model_dict = self._model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self._optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def _record_stats(self, train_loss, val_loss,val_metric):
        #import pdb; pdb.set_trace()
        self._training_losses.append(train_loss)
        self._val_losses.append(val_loss)
        self._val_metric.append(float(val_metric))
        self.plot_stats()

        write_to_file_in_dir(self._experiment_dir, 'training_losses.txt', self._training_losses)
        write_to_file_in_dir(self._experiment_dir, 'val_losses.txt', self._val_losses)
        write_to_file_in_dir(self._experiment_dir, 'val_metric.txt', self._val_metric)

    def _log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self._experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self._experiment_dir, file_name, log_str)

    def _log_epoch_stats(self, start_time):
        time_elapsed = datetime.datetime.now() - start_time
        time_to_completion = time_elapsed * (self._epochs - self._current_epoch - 1)
        train_loss = self._training_losses[self._current_epoch]
        val_loss = self._val_losses[self._current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self._current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self._log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self._training_losses)
        x_axis = np.arange(1, e + 1, 1)
        # Validation and training loss
        fig, axs = plt.subplots(2, sharex=True)
        axs[0].plot(x_axis, self._training_losses, label="Training Loss")
        axs[0].plot(x_axis, self._val_losses, label="Validation Loss")
        axs[0].set_ylabel("Cross-entropy")
        axs[0].set_title(self._name + " Training and validation cross-entropy loss")
        axs[0].legend()
        # Metric plot
        axs[1].plot(x_axis, self._val_metric, label="Val acc")
        axs[1].set_ylabel("Accuracy")
        axs[1].set_title(self._name + " Validation prediction accuracy")
        #plt.show()
        for ax in axs.flat:
            ax.set(xlabel='Epochs')
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.show()
        fig.savefig(os.path.join(self._experiment_dir, "stat_plot.png"))
        
    def pred_probs(self, images, batch_size = 32):
        transform_val_test = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Resize([512, 512]),
            transforms.Normalize(mean = self.mean_train, std = self.sd_train)
            ]
        )
        dataloader = DataLoader(numpy_datset(images = images, transform = transform_val_test), 
                              batch_size = batch_size, shuffle = False)
        self._model.eval()
        pred_probs = []
        for i, images in enumerate(dataloader):                
            images = images.to(self.device)
            pred_probs.append(self._model.pred_prob(images).detach().cpu().numpy())
        pred_probs = np.concatenate(pred_probs, axis = 0)
        return pred_probs.astype(float)
        
        
            
            