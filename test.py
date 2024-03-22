import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from model import SCMNet


# check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

def NormalizeData_np(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    # y = TPR - FPR
    y = TPR + (1 - FPR)
    # print(y)
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


all_bestACER = 100.0
followingAUC = 100.0
bestAPCER = 100.0
bestBPCER = 100.0
    
model_path = f"/shared2/SCMNet/"

test_live_data = np.load(f"/shared3/data_intra_Oulu_split_label/protocol_1/test_images_live_p1.npy")
test_spoof_data = np.load(f"/shared3/data_intra_Oulu_split_label/protocol_1/test_images_spoof_p1.npy")
    
print("Test")

print(test_live_data.shape, test_spoof_data.shape)

labels1 = np.ones(test_live_data.shape[0])
labels0 = np.zeros(test_spoof_data.shape[0])

test_images = np.vstack([test_live_data, test_spoof_data]) 
test_labels = np.hstack([labels1, labels0])

test_images_tensor = (torch.tensor(test_images).permute(0, 3, 1, 2).float())  # NHWC to NCHW
test_images_tensor = NormalizeData_torch(test_images_tensor)
test_labels_tensor = torch.tensor(test_labels).float()

test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

output_ACER = []
output_AUC = []


bestACER = 100.0

import glob 
paths = glob.glob(model_path + '/SCMNet/*.pth')
print(len(paths))

for i in range(len(paths)):
    model = SCMNet().to(device)
    model.load_state_dict(torch.load(model_path + f'/SCMNet/model_count_{i+1:03d}.pth'))
    model.eval()
    all_scores = []
    new_labels = []

    with torch.no_grad():
        cnt = 0
        for images, labels in test_loader:
            images = images.to(device)
            spoof_cue = model(NormalizeData_torch(images))

            # sum the maps of every pixel and normalize it with product of height and width
            scores_cues = torch.sum(spoof_cue, dim=(2, 3)) / (spoof_cue.shape[2] * spoof_cue.shape[3])
            scores_cues = torch.squeeze(scores_cues, 1)

            for k in range(0, spoof_cue.size(0)):
                
                all_scores.append(1.0 * scores_cues[k].cpu().numpy())
                new_labels.append(labels[k].cpu().numpy())
                

    fpr, tpr, thresholds = roc_curve(new_labels, all_scores, pos_label=1)
    threshold_cs, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)


    TP = 0.0000001
    TN = 0.0000001
    FP = 0.0000001
    FN = 0.0000001

    for j in range(len(all_scores)):
        score = all_scores[j]
        if (score >= threshold_cs and new_labels[j] == 1):
            TP += 1
        elif (score < threshold_cs and new_labels[j] == 0):
            TN += 1
        elif (score >= threshold_cs and new_labels[j] == 0):
            FP += 1
        elif (score < threshold_cs and new_labels[j] == 1):
            FN += 1

    APCER = FP / (TN + FP)
    NPCER = FN / (FN + TP)
    # print(f"ACER for classifier: {np.round((APCER + NPCER) / 2, 4)}")
    if np.round((APCER + NPCER) / 2, 4) < bestACER:
        bestACER = np.round((APCER + NPCER) / 2, 4)
        followingAUC = np.round(roc_auc_score(new_labels, all_scores), 4)
        bestAPCER = APCER
        bestBPCER = NPCER
    print(
        f"Epoch  {i+1:03d} ACER: {np.round((APCER + NPCER) / 2, 4)} AUC: {np.round(roc_auc_score(new_labels, all_scores),4)}")


print(f"best ACER: {bestACER} AUC {followingAUC} APCER {bestAPCER} BPCER {bestBPCER}")