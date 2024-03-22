import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import glob
from einops import rearrange

from model import SCMNet
from generateSCMap import GenerateSCMap_poly
from memoryBank import MemoryBank

# check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

def gram_schmidt(vectors):

    # Get the number of vectors
    num_vectors = vectors.size(0)
    
    # Initialize an empty tensor to store the orthogonalized vectors
    orthogonalized = torch.empty_like(vectors)
    
    for i in range(num_vectors):
        # Start with the original vector
        v = vectors[i, :].clone()
        
        # Subtract the projection onto the previously computed orthogonal vectors
        for j in range(i):
            v = v - (v @ orthogonalized[j, :]) * orthogonalized[j, :]
        
        # Normalize the result to obtain the next orthogonal vector
        orthogonalized[i, :] = v / torch.norm(v)
    
    return orthogonalized


model_path = f"/SCMNet/"

files = glob.glob(model_path + '/SCMNet/*.pth')
for f in files:
    os.remove(f)

# casia MSU Oulu replay
import torchvision.transforms as T
T_live = torch.nn.Sequential(
    T.RandomHorizontalFlip(),
) 

input_dataset1 = np.load(f"/shared3/domain-generalization/MSU_images_live.npy")
input_dataset2 = np.load(f"/shared3/domain-generalization/replay_images_live.npy")
input_dataset = np.concatenate((input_dataset1, input_dataset2), axis = 0)

batch_size = 8
learning_rate = 0.0005
num_epochs = 20
savecount = 0

L = int(input_dataset.shape[0] / batch_size) * batch_size
input_dataset = input_dataset[0: L]

live_dataset = torch.tensor(input_dataset).permute(0, 3, 1, 2).float()
live_dataset = NormalizeData_torch(live_dataset)

print(live_dataset.shape)
savestep = int(len(live_dataset)/ (batch_size * 4)) + 1
print("savestep",savestep)

real_label = np.ones(len(live_dataset), dtype=np.int64)
real_label = torch.tensor(real_label) 
trainset_D = torch.utils.data.TensorDataset(live_dataset,real_label)


dataloader = DataLoader(trainset_D, batch_size=batch_size, shuffle=True)

# Model
model = SCMNet().to(device)

# Loss and optimizer
MSE = nn.MSELoss()
COS = nn.CosineSimilarity(dim = 0)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
MB_size = 16
MB_partial = MemoryBank(size = MB_size)

cos_threshold = 0.2
sim_list = []

for epoch in range(num_epochs):
    
    step = 0
    for data in dataloader: 
        
        images, live_label = data
        images = images.to(device) 
        images = T_live(images)
        live_label = live_label.to(device)
        s = images.shape[0]
        step = step + 1

        m_s_partial_GT = torch.tensor(GenerateSCMap_poly(s, 32, 32), dtype=torch.float32).to(device)
        m_s_partial_GT = rearrange(m_s_partial_GT, 's h w -> s 1 h w')
        
        spoof_label = live_label * 0

        live_feature, partial_spoof_z, live_map, m_partial = model(NormalizeData_torch(images), m_s_partial_GT, update="learn_FE")
        
        map_loss_l = MSE(live_map, live_map * 0.0)
        
        partial_m_loss = MSE(m_partial, m_s_partial_GT)
 
        loss_ex_g = 1e-2 * map_loss_l + 1e-2 * partial_m_loss 

        # Backprop and optimize
        optimizer.zero_grad()
        loss_ex_g.backward(retain_graph=True)
        optimizer.step()

        if step % savestep == 0:
            print(f"Epoch {epoch:03d} Step {step:02d} FE step")

        m_s_partial_GT = torch.tensor(GenerateSCMap_poly(s), dtype=torch.float32).to(device)
        m_s_partial_GT = rearrange(m_s_partial_GT, 's h w -> s 1 h w')

        live_feature, partial_spoof_z, live_map, m_partial = model(
            NormalizeData_torch(images), m_s_partial_GT, update="learn_Gtr")

        partial_m_loss = MSE(m_partial, m_s_partial_GT)   

        # flatten all the features
        live_feature = rearrange(live_feature, 's c h w -> s (c h w)')
        partial_spoof_z = rearrange(partial_spoof_z, 's c h w -> s (c h w)')

        cos_loss_a_partial = 0

        if MB_partial.empty() != 1:
            mb_feature = MB_partial.get_memory()
            for l in range(mb_feature.shape[0]):
                for m in range(partial_spoof_z.shape[0]):
                    cos_loss_a_partial += abs(torch.mean(COS(mb_feature[l], partial_spoof_z[m])))
            cos_loss_a_partial = cos_loss_a_partial / (mb_feature.shape[0] * partial_spoof_z.shape[0])

        loss_g = 1e-2 * partial_m_loss + cos_loss_a_partial

        if MB_partial.get_length() < MB_size:
            prev_partial_spoof_z = partial_spoof_z.clone().detach()
            MB_partial.add(prev_partial_spoof_z[0])
        else:
            prev_partial_spoof_z = partial_spoof_z.clone().detach()
            mb_feature = MB_partial.get_memory()
            
            total_sim = 0
            for m in range(prev_partial_spoof_z.shape[0]):
                sim = 0
                for l in range(mb_feature.shape[0]):
                    sim += abs(torch.mean(COS(mb_feature[l], prev_partial_spoof_z[m])))

                sim = sim / mb_feature.shape[0]
                total_sim += sim

                if sim <= cos_threshold:
                    MB_partial.add(prev_partial_spoof_z[m])
        
                    features = gram_schmidt(MB_partial.get_memory())
                    for l in range(features.shape[0]):
                        MB_partial.pop()
                        MB_partial.add(features[l])

            
            avg_sim = total_sim / prev_partial_spoof_z.shape[0]
            sim_list.append(avg_sim.item())


        if step % savestep == 0:
            print(f"Epoch {epoch:03d} Step {step:02d} G")
        # Backprop and optimize
        optimizer.zero_grad()
        loss_g.backward()
        optimizer.step()
        
        if step % savestep == 0:
            print("Save model")
            if not os.path.exists(model_path + '/SCMNet/'):
                os.makedirs(model_path + '/SCMNet/') 
            torch.save(model.state_dict(), model_path + f'/SCMNet/model_count_{savecount+1:03d}.pth')
            savecount += 1


    print("Save model")
    if not os.path.exists(model_path+'/SCMNet/'): 
        os.makedirs(model_path+'/SCMNet/') 
    
    torch.save(model.state_dict(), model_path + f'/SCMNet/model_count_{savecount+1:03d}.pth')
    savecount += 1 
