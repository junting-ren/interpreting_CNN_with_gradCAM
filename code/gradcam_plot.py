from data_loader import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def plot_cam(exp, file_path, img, index, slice_type):
    pred = exp._model(img)
    pred[index].backward()
    # pull the gradients out of the model
    gradients = exp._model.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = exp._model.get_activations(img).detach()

    # weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap.cpu(), 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # # draw the heatmap
    # plt.matshow(heatmap.squeeze())
    import cv2
    heatmap = heatmap.cpu().numpy()
    img = cv2.imread(file_path[0])
    img = cv2.resize(img, (250, 250))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    if index==0:
        activation = 'horizontal'
    elif index==1:
        activation = 'coronally'
    else:
        activation = 'sagittal'
    os.makedirs('./gradcam', exist_ok=True)
    cv2.imwrite('./gradcam/'+slice_type+'_'+activation+'.jpg', superimposed_img)
    return pred

def plot_grad_cam(exp, plot_dataloader, slice_type = 'horizontal'):
    exp._model.eval()
    # get the image from the dataloader
    img, label, file_path = next(iter(plot_dataloader))
    img = img[:1,:,:].to(exp.device)
    label = label.to(exp.device)    
    #print(label)
    # loop over the three class activation regardless of the true class
    for i in range(3):
        preds = plot_cam(exp, file_path, img,i, slice_type)
    return np.round(torch.nn.functional.softmax(preds, dim = 0).cpu().detach().numpy(),4)


def help_plot(probs , plot_type = 'max', slice_name = 'horizontal' ):
    img0 = plt.imread('./gradcam/'+plot_type+'_entropy_'+slice_name+'_horizontal.jpg')
    img1 = plt.imread('./gradcam/'+plot_type+'_entropy_'+slice_name+'_coronally.jpg')
    img2 = plt.imread('./gradcam/'+plot_type+'_entropy_'+slice_name+'_sagittal.jpg')
    plt.figure()

    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1,3) 
    f.suptitle('Correct class: '+ slice_name+ ' ' + plot_type + ' entropy example', y=0.8)
    
    axarr[0].imshow(img0)
    axarr[0].title.set_text('h_prob='+str(probs[0]))
    axarr[1].imshow(img1)
    axarr[1].title.set_text('c_prob='+str(probs[1]))
    axarr[2].imshow(img2)
    axarr[2].title.set_text('s_prob='+str(probs[2]))
    #f.tight_layout()
    f.savefig('./gradcam/'+plot_type+'_'+slice_name+'_activation.jpg')

def show_gradcam(exp, true_labels, cross_entropy_l, file_paths_l):
    transform_val_test = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Resize([512, 512]),
        transforms.Normalize(mean = exp.mean_train, std = exp.sd_train)
        ]
    )
    index_horizontal = np.where([0==x for x in true_labels])[0]
    index_coronally = np.where([1==x for x in true_labels])[0]
    index_sagittal = np.where([2==x for x in true_labels])[0]
    cross_entropy_horizontal = np.array(cross_entropy_l)[index_horizontal]
    cross_entropy_coronally = np.array(cross_entropy_l)[index_coronally]
    cross_entropy_sagittal = np.array(cross_entropy_l)[index_sagittal]
    file_paths_horizontal =  np.array(file_paths_l)[index_horizontal]
    file_paths_coronally =  np.array(file_paths_l)[index_coronally]
    file_paths_sagittal =  np.array(file_paths_l)[index_sagittal]

    index_max_horizontal = cross_entropy_horizontal == max(cross_entropy_horizontal)
    index_max_coronally = cross_entropy_coronally == max(cross_entropy_coronally)
    index_max_sagittal = cross_entropy_sagittal == max(cross_entropy_sagittal)

    file_max_horizontal = file_paths_horizontal[index_max_horizontal][0]
    file_max_coronally = file_paths_coronally[index_max_coronally][0]
    file_max_sagittal = file_paths_sagittal[index_max_sagittal][0]

    index_min_horizontal = cross_entropy_horizontal == min(cross_entropy_horizontal)
    index_min_coronally = cross_entropy_coronally == min(cross_entropy_coronally)
    index_min_sagittal = cross_entropy_sagittal == min(cross_entropy_sagittal)

    file_min_horizontal = file_paths_horizontal[index_min_horizontal][0]
    file_min_coronally = file_paths_coronally[index_min_coronally][0]
    file_min_sagittal = file_paths_sagittal[index_min_sagittal][0]
    
    index_random_horizontal = np.random.randint(low = 0, high = len(cross_entropy_horizontal))
    index_random_coronally = np.random.randint(low = 0, high = len(cross_entropy_coronally))
    index_random_sagittal = np.random.randint(low = 0, high = len(cross_entropy_sagittal))
    
    #import pdb; pdb.set_trace()
    file_random_horizontal = file_paths_horizontal[index_random_horizontal]
    file_random_coronally = file_paths_coronally[index_random_coronally]
    file_random_sagittal = file_paths_sagittal[index_random_sagittal]


    plot_dataloader_horizontal_max = DataLoader(brain_image_datset([file_max_horizontal],  torch.Tensor([0]), transform = transform_val_test), batch_size = 1)
    plot_dataloader_coronally_max = DataLoader(brain_image_datset([file_max_coronally],  torch.Tensor([1]), transform = transform_val_test), batch_size = 1)
    plot_dataloader_sagittal_max = DataLoader(brain_image_datset([file_max_sagittal],  torch.Tensor([2]), transform = transform_val_test), batch_size = 1)

    plot_dataloader_horizontal_min = DataLoader(brain_image_datset([file_min_horizontal],  torch.Tensor([0]), transform = transform_val_test), batch_size = 1)
    plot_dataloader_coronally_min = DataLoader(brain_image_datset([file_min_coronally],  torch.Tensor([1]), transform = transform_val_test), batch_size = 1)
    plot_dataloader_sagittal_min = DataLoader(brain_image_datset([file_min_sagittal],  torch.Tensor([2]), transform = transform_val_test), batch_size = 1)
    
    plot_dataloader_horizontal_random = DataLoader(brain_image_datset([file_random_horizontal],  torch.Tensor([0]), transform = transform_val_test), batch_size = 1)
    plot_dataloader_coronally_random = DataLoader(brain_image_datset([file_random_coronally],  torch.Tensor([1]), transform = transform_val_test), batch_size = 1)
    plot_dataloader_sagittal_random = DataLoader(brain_image_datset([file_random_sagittal],  torch.Tensor([2]), transform = transform_val_test), batch_size = 1)
    
    preds_h_max = plot_grad_cam(exp, plot_dataloader_horizontal_max, slice_type = 'max_entropy_horizontal')
    preds_c_max = plot_grad_cam(exp, plot_dataloader_coronally_max, slice_type = 'max_entropy_coronally')
    preds_s_max = plot_grad_cam(exp, plot_dataloader_sagittal_max, slice_type = 'max_entropy_sagittal')

    preds_h_min = plot_grad_cam(exp, plot_dataloader_horizontal_min, slice_type = 'min_entropy_horizontal')
    preds_c_min = plot_grad_cam(exp, plot_dataloader_coronally_min, slice_type = 'min_entropy_coronally')
    preds_s_min = plot_grad_cam(exp, plot_dataloader_sagittal_min, slice_type = 'min_entropy_sagittal')
    
    preds_h_random = plot_grad_cam(exp, plot_dataloader_horizontal_random, slice_type = 'random_entropy_horizontal')
    preds_c_random = plot_grad_cam(exp, plot_dataloader_coronally_random, slice_type = 'random_entropy_coronally')
    preds_s_random = plot_grad_cam(exp, plot_dataloader_sagittal_random, slice_type = 'random_entropy_sagittal')
    
    help_plot(preds_h_max,plot_type = 'max', slice_name = 'horizontal')
    help_plot(preds_c_max,plot_type = 'max', slice_name = 'coronally')
    help_plot(preds_s_max,plot_type = 'max', slice_name = 'sagittal')
    
    help_plot(preds_h_min,plot_type = 'min', slice_name = 'horizontal')
    help_plot(preds_c_min,plot_type = 'min', slice_name = 'coronally')
    help_plot(preds_s_min,plot_type = 'min', slice_name = 'sagittal')
    
    help_plot(preds_h_random,plot_type = 'random', slice_name = 'horizontal')
    help_plot(preds_c_random,plot_type = 'random', slice_name = 'coronally')
    help_plot(preds_s_random,plot_type = 'random', slice_name = 'sagittal')