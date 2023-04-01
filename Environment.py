import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import models.Classifier.pretrained_CNN as pretrained_CNN

import os

import numpy as np

# batch_size = 6000

def pos(x):
    cen_ar =[]
    for j in range(len(x)):
        cen = [14,14]
        x_t = x[j].item()
        # print(x_t)
        temp = x_t -756
        for i in range(1,28):
            # print(i)
            
            if i % 2 != 0:
                # print(x_t-i<0)
                if x_t - i < 0 :
                    
                    cen[0] = cen[0] - x_t
                    break
                x_t = x_t - i
                # print(x)
                cen[0] = cen[0] - i
                if x_t - i < 0 :
                    
                    cen[1] = cen[1] - x_t 
                    break
                x_t = x_t - i
                
                cen[1] = cen[1] - i

            else : 
                if x_t - i < 0 :
                
                    cen[0] = cen[0] + x_t 
                    break
                x_t = x_t - i
                
                cen[0] = cen[0] + i
                if x_t - i < 0 :
                    
                    cen[1] = cen[1] + x_t 
                    break
                x_t = x_t - i
                # print(x)
                cen[1] = cen[1] + i
        # print(temp)
        if temp > 0:
            cen[0] = cen[0] + temp

        cen_ar.append(cen)
        
    return cen_ar




class MnistEnv():
    def __init__(self, classification_model):
        super().__init__()
        self.classification_model = classification_model

    def make_transformed_images(self, original_images, actions):
        # actions = torch.tanh(actions)
        # arr = []
        # x, y, brightness = actions[:, 0], actions[:, 1], actions[:, 2]
        # x = (13.5+13.5*x ).int()
        # y = (13.5+13.5*y ).int()
        # brightness = (brightness * 255).round()/255
        batch_size = original_images.shape[0]
        actions = torch.sigmoid(actions)
        arr = []
        x = (actions[:, 0] * 27).int()
        y = (actions[:, 1] * 27).int()
        brightness = ((actions[:, 2] * 255).int()) / 255
        x, y, brightness = actions[:, 0], actions[:, 1], actions[:, 2]
        x = (x * 27).int()
        y = (y * 27).int()
        # print(x,y)
        brightness = (brightness * 255).round()/255
        for i in range(batch_size):
            changed_image = original_images[i].squeeze(
            ).squeeze().detach().cpu().numpy()
            changed_image[x[i], y[i]] = brightness[i]
            arr.append(changed_image)
        changed_images = torch.stack(
            [torch.tensor(a).unsqueeze(0) for a in arr], dim=0)

        return changed_images
    
    def make_transformed_images_3d(self, original_images, actions):
        # actions = torch.tanh(actions)
        # arr = []
        # x, y, brightness = actions[:, 0], actions[:, 1], actions[:, 2]
        # x = (13.5+13.5*x ).int()
        # y = (13.5+13.5*y ).int()
        # brightness = (brightness * 255).round()/255
        batch_size = original_images.shape[0]
        action = torch.sigmoid(actions)  # 0~1 사이로 만들어줌
        point = (action[:, 0] * 783).int()
        brightness = ((action[:, 1] * 255).int()) / 255

        arr = []
        center = [point//28, point % 28]
        action = [point, brightness]
        
        for i in range(batch_size):
            changed_image = original_images[i].squeeze().squeeze().detach().cpu().numpy()  #   [28,28]
            changed_image[center[0][i], center[1][i]] = brightness[i]
            arr.append(changed_image)

        changed_images = torch.stack([torch.tensor(a).unsqueeze(0) for a in arr], dim=0)

        return changed_images
    

    def make_transformed_images_circle(self, original_images, actions):
        batch_size = original_images.shape[0]
        actions = torch.sigmoid(actions)
        arr = []
        cen = pos((783*actions.cpu()).int())
        cen = np.array(cen)
        for i in range(batch_size):
            changed_image = original_images[i].squeeze(
            ).squeeze().detach().cpu().numpy()
            # print(cen[i])
            changed_image[cen[i][0],cen[i][1]] = 1
            arr.append(changed_image)
        changed_images = torch.stack(
            [torch.tensor(a).unsqueeze(0) for a in arr], dim=0)

        return changed_images



    def step(self, original_images, actions, device, model_type):
        batch_size = original_images.shape[0]
        original_images = original_images.view(-1,1,28,28)
        if model_type == 'learnable_pos' or model_type =='heat_learn' or model_type == 'normal'or model_type=='learnable_cat':

            changed_images = self.make_transformed_images(original_images, actions)

        
        
        elif model_type == 'x_y_linear' or model_type =='heat_linear' :
            changed_images = self.make_transformed_images_3d(original_images, actions)
        
        elif model_type == 'circle':
            changed_images = self.make_transformed_images_circle(original_images, actions)

        with torch.no_grad():
            original_outputs = self.classification_model(
                original_images.to(device))
            changed_outputs = self.classification_model(
                changed_images.to(device))


        ########################################################### label reward #########################################
        p = torch.argmax(original_outputs, dim=1)
        c = torch.argmax(changed_outputs, dim=1)
        temp = original_outputs - changed_outputs
        rewards = torch.zeros(len(temp)).to(device)
        count = 0
        for i in range(len(p)):
            # rewards[i] = 1000*temp[i,p[i]]
            rewards[i] = 0
            if c[i] != p[i]:
                count +=1 
                rewards[i] = 1000

        ########################################################### kld reward #########################################
        # rewards = torch.sum(
        #     torch.nn.functional.kl_div(
        #         changed_outputs.log(), original_outputs, size_average=None, reduction="none"
        #     ),
        #     dim=1,
        # )
        
        
        # rewards = torch.where(rewards < 0, 0.0001,
        #                       rewards)  # 음수일 경우 0.0001로 바꿔줌
        # rewards = torch.where(
        #     torch.argmax(original_outputs, dim=1) == torch.argmax(
        #         changed_outputs, dim=1),
        #     -1 / rewards,
        #     rewards,
        # )   # 같은 클래스일 경우 -1/reward, 다른 클래스일 경우 reward

        
        # rewards = rewards.cpu()

        # rewards[rewards < -1000] = -1000
        # rewards[rewards > 1000] = 1000
        # rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1000, neginf=-1000)

        # correct = torch.zeros(batch_size, device=device)
         
        ############################################################ daehwa's label #########################################
       

        # rewards = torch.where(p == c,-1000.0, 1000.0)
        # rewards = torch.where(label != original_preds, 0,
        #                       rewards)  # Noise Reduction
        
        return rewards.cpu().numpy() ,count
    
    def step_circle(self, original_images, actions, device):
        changed_images = self.make_transformed_images_circle(original_images, actions)
        with torch.no_grad():
            original_outputs = self.classification_model(
                original_images.to(device))
            changed_outputs = self.classification_model(
                changed_images.to(device))

        rewards = torch.sum(
            torch.nn.functional.kl_div(
                changed_outputs.log(), original_outputs, size_average=None, reduction="none"
            ),
            dim=1,
        )

        rewards = torch.where(rewards < 0, 0.0001,
                              rewards)  # 음수일 경우 0.0001로 바꿔줌
        rewards = torch.where(
            torch.argmax(original_outputs, dim=1) == torch.argmax(
                changed_outputs, dim=1),
            -1 / rewards,
            rewards,
        )   # 같은 클래스일 경우 -1/reward, 다른 클래스일 경우 reward

        rewards[rewards < -1000] = -1000
        rewards[rewards > 1000] = 1000
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1000, neginf=-1000)
        return rewards.cpu().numpy()

    def evaluate(self, original_images, label, actions, device, type):
        img=[]
        img_dif = []
        label_o = []
        label_c = []
        label_arr = []
        original_images = original_images.view(-1,1,28,28)
        if type == 'learnable_pos' or type =='heat_learn' or type == 'normal'or type=='learnable_cat':
            changed_images = self.make_transformed_images(original_images, actions)


        elif type == 'x_y_linear' or type =='heat_linear' :
            changed_images = self.make_transformed_images_3d(original_images, actions)

        elif type == 'circle':
            changed_images = self.make_transformed_images_circle(original_images, actions)

        self.classification_model.eval()
        with torch.no_grad():
            original_outputs = self.classification_model(
                original_images.to(device)).cpu()
            changed_outputs = self.classification_model(
                changed_images.to(device)).cpu()
        p_o = torch.argmax(original_outputs, dim=1)
        p_c = torch.argmax(changed_outputs, dim=1)
        
        
        idx = np.where((p_o != p_c) == 1)
        
        original_images = np.array(original_images.cpu())
        changed_images = np.array(changed_images.cpu())

        for i in idx[0]:
            # print(label[i].item(),p_o[i].item(),p_c[i].item())
            img.append(original_images[i])
            img_dif.append(changed_images[i])
            label_arr.append(label[i].item())
            label_o.append(p_o[i].item())
            label_c.append(p_c[i].item())
        different = torch.sum(torch.argmax(
            original_outputs, dim=1) != torch.argmax(changed_outputs, dim=1))
        return different.cpu().numpy(), img,img_dif, label_arr,label_o, label_c
    
    def evaluate_circle(self, original_images, label, actions, device):
        img=[]
        img_dif = []
        label_o = []
        label_c = []
        label_arr = []
        changed_images = self.make_transformed_images_circle(original_images, actions)
        self.classification_model.eval()
        with torch.no_grad():
            original_outputs = self.classification_model(
                original_images.to(device)).cpu()
            changed_outputs = self.classification_model(
                changed_images.to(device)).cpu()
        p_o = torch.argmax(original_outputs, dim=1)
        p_c = torch.argmax(changed_outputs, dim=1)
        
        
        idx = np.where((p_o != p_c) == 1)
        
        original_images = np.array(original_images.cpu())
        changed_images = np.array(changed_images.cpu())

        for i in idx[0]:
            print(label[i].item(),p_o[i].item(),p_c[i].item())
            img.append(original_images[i])
            img_dif.append(changed_images[i])
            label_arr.append(label[i].item())
            label_o.append(p_o[i].item())
            label_c.append(p_c[i].item())
        different = torch.sum(torch.argmax(
            original_outputs, dim=1) != torch.argmax(changed_outputs, dim=1))
        return different.cpu().numpy(), img,img_dif, label_arr,label_o, label_c
    
    def save_different_images(self, original_images, actions, device):
        changed_images = self.make_transformed_images_circle(original_images, actions)
        with torch.no_grad():
            original_outputs = self.classification_model(
                original_images.to(device))
            changed_outputs = self.classification_model(
                changed_images.to(device))

        different_indices = torch.argmax(
            original_outputs, dim=1) != torch.argmax(changed_outputs, dim=1)
        original_different_images = original_images[different_indices]
        changed_different_images = changed_images[different_indices]
        different_actions = actions[different_indices]

        for i, (ori_img, changed_img, action) in enumerate(zip(original_different_images, changed_different_images, different_actions)):
            label = torch.argmax(original_outputs[i]).item()
            folder_name = "label_{}".format(label)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                os.makedirs("{}/original".format(folder_name))
                os.makedirs("{}/changed".format(folder_name))
                os.makedirs("{}/action".format(folder_name))
            np.save("{}/original/{}.npy".format(folder_name, i), ori_img.cpu().numpy())
            np.save("{}/changed/{}.npy".format(folder_name, i), changed_img.cpu().numpy())
            np.save("{}/action/{}_action.npy".format(folder_name, i), action.cpu().numpy())
