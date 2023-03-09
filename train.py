import pylab
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from load_ACDC import ACDC_dataset



# iou计算
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


traindata, testdata = ACDC_dataset()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = ProbabilisticUnet(input_channels=1, num_classes=4, num_filters=[48, 96, 192, 384, 768], latent_dim=2, no_convs_fcomb=4,
                        beta=10.0)
net.to(device)

# 加载权重训练
# state_dict = torch.load('./weights/epoch_29,loss_0.0023,Iou_0.2107,test_Iou_0.216.pth')
# net.load_state_dict(state_dict)


optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
epochs = 50
plt_iou = []


for epoch in range(epochs):
    epoch_train_iou = []
    net.train()
    for step, (patch, mask) in enumerate(tqdm(traindata)):
        patch, mask = patch.to(device), mask.to(device)
        mask = torch.unsqueeze(mask, 1)
        net.forward(patch, mask, training=True)
        elbo, y_pred = net.elbo(mask)
        reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
        loss = -elbo + 1e-5 * reg_loss
        # loss = loss_fn(y_pred, torch.squeeze(mask).type(torch.long))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            batch_iou = iou_mean(y_pred, mask, 4)
            epoch_train_iou.append(batch_iou)


    epoch_test_iou = []
    net.eval()
    with torch.no_grad():
        for step, (patch, mask) in enumerate(tqdm(testdata)):
            patch, mask = patch.to(device), mask.to(device)
            mask = torch.unsqueeze(mask, 1)
            net.forward(patch, mask, training=True)
            elbo, y_pred = net.elbo(mask)

            y_pred = torch.argmax(y_pred, dim=1)
            batch_test_iou = iou_mean(y_pred, mask, 4)
            epoch_test_iou.append(batch_test_iou)



    print('epoch:', epoch,
          'train_Iou:', round(np.mean(epoch_train_iou), 4),
          'test_Iou:', round(np.mean(epoch_test_iou), 4)
          )
    plt_iou.append(round(np.mean(epoch_test_iou), 4))
    static_dict = net.state_dict()
    torch.save(static_dict, './weights/epoch_{},Iou_{},test_Iou_{}.pth'
               .format(epoch,
                       round(np.mean(epoch_train_iou), 4),
                       round(np.mean(epoch_test_iou), 4)
                       ))

plt.plot(range(1, epochs + 1), plt_iou, label='Iou')
plt.legend()
plt.savefig('./Iou_figure.png')
pylab.show()
