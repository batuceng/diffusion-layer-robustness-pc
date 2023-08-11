import torch
import numpy as np
from sklearn.metrics import accuracy_score
import os
from os.path import dirname, abspath
import random

def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
class Attack(object):
    def __init__(self, name, model, device, seed):
        self.model = model
        self.model.to(device).eval()
        self.name = name
        self.seed = seed
        self.device = device
        if seed is not None: seed_all(seed)
        
    def attack(self):
        raise NotImplementedError
        
    def get_logits(self, inputs):
        logits = self.model(inputs.permute(0,2,1).to(self.device))
        return logits
        
    def save(self, dataloader, root=dirname(dirname(abspath(__file__)))):
        true_labels, clean_preds, attack_preds = [], [], []
        attacked_batches = []
        
        for i,batch in enumerate(dataloader):
            print(f"batch {i}/{len(dataloader)}")
            pc, label = batch['pointcloud'], batch['cate']
            # Forward
            logits = self.get_logits(pc)
            adv_data = self.attack(data=pc, labels=label)
            # Check attack
            attacked_logits = self.get_logits(adv_data)
            # Store for Acc Stats
            true_labels.append(label.squeeze().detach().cpu().numpy())
            clean_preds.append(logits.detach().cpu().numpy().argmax(axis=1))
            attack_preds.append(attacked_logits.detach().cpu().numpy().argmax(axis=1))
            # Store Attacked Data
            batch['attacked'] = adv_data.detach().cpu()
            if not self.targeted:           # Save target too
                batch['target'] = torch.ones_like(batch['cate']) * -1
            else: 
                raise NotImplementedError
            # Add each instance to total list
            attacked_batches.extend([{key:batch[key][i] for key in batch} for i in range(pc.shape[0])])
            
        print("Clean Acc:",accuracy_score(np.concatenate(true_labels), np.concatenate(clean_preds)))
        print("Attacked Acc:",accuracy_score(np.concatenate(true_labels), np.concatenate(attack_preds)))
        
        # Write to root
        if root != None:
            DATA_DIR = os.path.join(root, 'data_attacked')
            if not os.path.exists(DATA_DIR):
                os.mkdir(DATA_DIR)
            datasetname, modelname = dataloader.dataset.__class__.__name__, self.model.__class__.__name__
            FILE_NAME = os.path.join(DATA_DIR, f'{datasetname}_{modelname}_{self.name}_eps_{self.eps}.pt')
            torch.save(attacked_batches,f=FILE_NAME)
        return attacked_batches
