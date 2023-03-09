import pylab
from matplotlib import pyplot as plt
from probabilistic_unet import ProbabilisticUnet
from load_ACDC import ACDC_dataset
import torch
import numpy as np

def iou_mean(pred, target, n_classes=4):
    # n_classes:the number of classes in your dataset,not including background
    ious = []
    iousSum = 0
    pred = pred.view(-1)
    target = np.array(target.cpu())
    target = torch.from_numpy(target)
    target = target.view(-1)
    # Ignore Iou for background class("0")
    for cls in range(1, n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            iousSum += float(intersection) / float(max(union, 1))
    return iousSum / (n_classes - 1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


traindata, testdata = ACDC_dataset()

Iou = []

def Predict_result():
    global model, img, label, state_dict, elbo
    model = ProbabilisticUnet(input_channels=1, num_classes=4, num_filters=[48, 96, 192, 384, 768], latent_dim=2,
                              no_convs_fcomb=4, beta=10.0)
    model.to('cuda')
    img, label = next(iter(testdata))
    img = img.to('cuda')
    label = label.to('cuda')
    label = torch.unsqueeze(label, 1)
    state_dict = torch.load('./weights/epoch_49,Iou_0.9269,test_Iou_0.8656.pth')
    model.load_state_dict(state_dict)
    model.eval()
    model.forward(img, label)
    elbo, pred = model.elbo(label)
    pred = torch.argmax(pred, dim=1)
    pred = torch.unsqueeze(pred, dim=1)
    for i in range(pred.shape[0]):
        if np.max(label[i].cpu().numpy()) > 0 and np.max(pred[i].cpu().numpy()) > 0:
            batch_iou = iou_mean(pred, label, 4)
            Iou.append(batch_iou)

        plt.figure(figsize=(12, 12))
        plt.subplot(1, 3, 1)
        plt.imshow(img[i].permute(1, 2, 0).cpu().numpy())
        plt.subplot(1, 3, 2)
        plt.imshow(label[i].permute(1, 2, 0).cpu().numpy())
        plt.subplot(1, 3, 3)
        plt.imshow(pred[i].permute(1, 2, 0).cpu().detach().numpy())
        pylab.show()



if __name__ == '__main__':
    Predict_result()
    print(round(np.mean(Iou), 4))
