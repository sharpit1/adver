import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.distributions as dist
import torch.optim as optim
import numpy as np
import torch.nn as nn
import wandb
import torch.nn.functional as F
import random
import models.Classifier.pretrained_CNN as pretrained_CNN
import models.RL.Adversarial_RL_x_y_linear as Adversarial_RL_x_y_linear
import models.RL.Adversarial_RL_learnable_pos as Adversarial_RL_learnable_pos
import models.RL.Adversarial_RL_heat_learn as Adversarial_RL_heat_learn
import models.RL.Adversarial_RL_heat_linear as Adversarial_RL_heat_linear
import models.RL.Adversarial_RL_circle as Adversarial_RL_circle
from models.Xai.explain import ADV,ex_class
import Environment
import models.train_function as train_function
import traceback



is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

print('Current cuda device is', device)

# Hyperparameters
config = {
    'action_dim': 3,
    'env_name': 'MnistEnv',
    'agent_name': 'REINFORCE',
    'batch_size': 6000,
    'model': 'x_y_linear',    # learnable_pos, x_y_linear, heat_learn, heat_linear, circle, normal, learnable_cat
    'RL_epoch': 200,
    'reward_type': 'minus_one_to_one',  # minus_one_to_one, zero_to_one
}

# wandb
sweep_config = {
    'method': 'grid',  # grid, random
    'metric': {
        'name': 'epoch_score',
        'goal': 'maximize'
    },
    'parameters': {
        'RL_learning_rate': {
            'values': [0.00001, 0.00002, 0.00004]
        },
        'action_std': {
            'values': [1]
        },
        'decay_rate': {
            'values': [0.999]
        }
    }
}


# Load MNIST dataset
train_data = datasets.MNIST(root='./MNIST_data/',
                            train=True,
                            download=True,
                            transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=config['batch_size'], shuffle=False
)

# set seed


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# sampling action


def sample_action(actions_mean, action_std, config):
    action_size = actions_mean.shape[1]
    cov_mat = torch.diag_embed(torch.full((action_size,), float(
        action_std ** 2))).to(config['device']).unsqueeze(0)
    distribution = dist.MultivariateNormal(actions_mean, cov_mat)
    actions = distribution.sample()
    actions_logprob = distribution.log_prob(actions)
    return actions, actions_logprob

# XAI model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode, dilation, groups, bias)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
        )
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding="same")
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(3136, 1000)  # 7 * 7 * 64 = 3136
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        return F.softmax(x, dim = 1)

cnn_weight = torch.load('./parameters/mnist_cnn.pt')
cnn = CNN()
cnn.load_state_dict(cnn_weight)
cnn = cnn.to(device)


class REINFORCE(nn.Module):
    def __init__(self):
        super(REINFORCE, self).__init__()
        self.data = []
        if config['model'] == 'learnable_pos' or config['model'] == 'normal' :
            self.conv1 = nn.Conv2d(1, 32, 3, 1, padding="same")
            self.action_mean = torch.nn.Linear(1024, 3)
        elif config['model'] == 'x_y_linear':
            self.conv1 = nn.Conv2d(3, 32, 3, 1, padding="same")
            self.action_mean = torch.nn.Linear(1024, 2)
        elif config['model'] == 'heat_learn':
            self.conv1 = nn.Conv2d(2, 32, 3, 1, padding="same")
            self.action_mean = torch.nn.Linear(1024, 3)
        elif config['model'] == 'heat_linear':
            self.conv1 = nn.Conv2d(4, 32, 3, 1, padding="same")
            self.action_mean = torch.nn.Linear(1024, 2)
        elif config['model'] == 'circle':
            self.conv1 = nn.Conv2d(1, 32, 3, 1, padding="same")
            self.action_mean = torch.nn.Linear(1024, 1)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding="same")

        self.fc1 = nn.Linear(3136, 1024)  # 7 * 7 * 64 = 3136
        self.fc2 = nn.Linear(1024, 1024)

        
        if config['model'] == 'learnable_cat':
            self.positions = nn.Parameter(torch.randn(config['batch_size'],1 ,28, 28)/10, requires_grad=True)
            self.conv1 = nn.Conv2d(2, 32, 3, 1, padding="same")
            self.action_mean = torch.nn.Linear(1024, 3)
        else:

            self.positions = nn.Parameter(torch.randn(28, 28)/10, requires_grad=True)

        self.optimizer = optim.Adam(self.parameters(), lr=0)
        self.action_var = (torch.ones(3)*0.5).to(device)

    def forward(self, image,pos=None):
        # print(pos.shape)
        # print(image.shape)
        if config['model'] == 'learnable_pos' or config['model'] == 'heat_learn' or config['model'] == 'circle':
            image = image + pos
        if config['model'] == 'learnable_cat':
            image = torch.cat([image,pos], dim=1)

        image = self.conv1(image)
        image = F.relu(image)
        image = F.max_pool2d(image, 2)
        image = self.conv2(image)
        image = F.relu(image)
        image = F.max_pool2d(image, 2)
        image = torch.flatten(image, 1)
        image = self.fc1(image)
        image = F.relu(image)
        image = self.fc2(image)
        image = F.relu(image)
        action_mean = self.action_mean(image)
        action_mean = torch.sigmoid(action_mean)
        self.action_var *= 0.99
        
        # if self.action_var[0] < 0.01:
        #     self.action_var = (torch.ones(3)*0.01).to(device)

        # dist = MultivariateNormal(action_mean, torch.diag(self.action_var))
        # action = dist.sample()
        # action_log_prob = dist.log_prob(action)
        return action_mean

policy = REINFORCE().to(device)
# train



def train():
    try:
        
        early_stopping_patience = 200
        best_score = -np.inf
        no_improvement = 0

        # wandb.init(project="test", entity="adversarial-rl",config=sweep_config_defaults)
        

        # wandb.run.name = 'feedback_'+config['model']+'_sweep'
        # wandb.run.save()
        # load RL model
        if config['model'] == 'learnable_pos':
            agent = Adversarial_RL_learnable_pos.REINFORCE(
                config).to(config['device'])
        elif config['model'] == 'normal':
            agent = Adversarial_RL_learnable_pos.REINFORCE_normal(
                config).to(config['device'])
        elif config['model'] == 'learnable_cat':
            agent = Adversarial_RL_learnable_pos.REINFORCE_cat(
                config).to(config['device'])
        elif config['model'] == 'x_y_linear':
            agent = Adversarial_RL_x_y_linear.REINFORCE(
                config).to(config['device'])
        elif config['model'] == 'heat_learn':
            agent = Adversarial_RL_heat_learn.REINFORCE(
                config).to(config['device'])
        elif config['model'] == 'heat_linear':
            agent = Adversarial_RL_heat_linear.REINFORCE(
                config).to(config['device'])
        elif config['model'] == 'circle':
            agent = Adversarial_RL_circle.REINFORCE(config).to(config['device'])

        else:
            raise ValueError(f"Invalid model name: {config['model']}")\
            
                   
        optimizer = torch.optim.Adam(
            agent.parameters(), lr=wandb.config.RL_learning_rate)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)


        # load CNN model and evaluate
        if epoch == 0:
            CNN = pretrained_CNN.CNN().to(config['device'])
            pretrained_CNN.evaluate(CNN, config['device'])
        
        elif epoch != 0:
            CNN = cnn.to(device)
            pretrained_CNN.evaluate(cnn, config['device'])

        CNN.eval()

        # load MNIST Environment
        env = Environment.MnistEnv(CNN)

        reward_set = []
        reward_last = torch.zeros(config['batch_size']).to(device)
        cen_last = torch.zeros(config['batch_size'],2).to(device)
        # criteria = []
        action_std = wandb.config.action_std
        wandb.watch(agent, log='all', log_freq=1) 
        print("RL_epoch:",config['RL_epoch'])
        for i in range(config['RL_epoch']):   # RL epoch 만큼
            dif = []

            
            # cen_last = cen_current
            epoch_score = 0  # score reset
            for s, _ in train_loader:
                s = s.to(config['device'])
                actions_mean = agent(s)
                actions, actions_log_prob = sample_action(
                    actions_mean, action_std, config)
                x = (torch.sigmoid(actions[:, 0]) * 27).int()
                y = (torch.sigmoid(actions[:, 1]) * 27).int()
                cen = torch.stack([x,y],dim=1)
                rewards, c = env.step(s[:,0].type(torch.float32),
                                   actions, config['device'],config['model'])
                epoch_score += np.sum(rewards)
                criteria = np.sum(rewards)/config['batch_size']
                reward_set.append(rewards)
                # print(cen[i])
                # print(cen_last[i])
                dif.append(c)
                # print(reward_last.shape)
                
                if criteria>7:

                    for n in range(config['batch_size']):
                        if rewards[n] != 1000:
                            if torch.equal(cen[n], cen_last[n]):
                                print(cen[n], cen_last[n])
                                agent.data.append(
                                    [-1000, actions_log_prob[n]]
                                )
                            else:agent.data.append([rewards[n], actions_log_prob[n]])
                        else:
                            
                                agent.data.append([rewards[n], actions_log_prob[n]])
                else:
                    for n in range(config['batch_size']):
                        
                        
            

                        agent.data.append([rewards[n], actions_log_prob[n]])
                optimizer.zero_grad()
                r_lst, log_prob_lst = zip(*agent.data)
                r_lst = torch.tensor(r_lst).to(config['device'])
                log_prob_lst = torch.stack(log_prob_lst).to(config['device'])
                loss = -log_prob_lst * r_lst
                # criteria = (reward_set == 1000)
                loss.mean().backward()
                optimizer.step()
                agent.data = []

                action_std *= sweep_config_defaults['decay_rate']

                cen_last = cen
                reward_last = rewards
                if action_std < 0.1:
                    action_std = 0.1
            
            wandb.log({"epoch_score": epoch_score / 60000 , "change": sum(dif)})
            # wandb.log({"change": sum(dif)})
            
            # scheduler.step()
            print("epoch_score: ", epoch_score/60000)
            print("change : ",sum(dif))

        torch.save(agent.state_dict(), "./Result/{}/model_".format(config['model'])+str(i)+'.pt')
        pretrained_CNN.evaluate(env.classification_model, config['device'])
        agent.eval()
        count = 0
        ori_arr = []
        change_arr = []
        label = []
        label_o_arr = []
        label_c_arr = []
        for s, q in train_loader:
                        s = s.to(config['device'])
                        actions_mean = agent(s)
                        actions, _ = sample_action(
                            actions_mean, action_std, config)
                        temp, ori, change_img, label_arr,label_o, label_c = env.evaluate(s[:,0].type(torch.float32),q,
                                            actions, config['device'], config['model'])
                        count += temp
                        # print(temp)
                        # print(len(change_img))
                        if temp > 0:
                            ori_arr.extend(ori)
                            change_arr.extend(change_img)
                            label.extend(label_arr)
                            label_o_arr.extend(label_o)
                            label_c_arr.extend(label_c)
                    # wandb.summary["confused_images"] = count
        wandb.log({"confused_images" : count})
        ex = []
        for i  in range(count):
            # print(change_img[i][0])
            # print(i)
            a = wandb.Image(np.array(change_arr[i][0]*255).astype(np.uint8), caption=f"Label: {label_c_arr[i]}")
            ex.append(a)
        wandb.log({"adversarial":ex})
        # wandb.join()
        print("confused : ",count)
        np.save('./Result/{}/RL_original_penalty_img_'.format(config['model'])+str(config['RL_epoch'])+'batch_size'+str(config['batch_size']),ori_arr)
        np.save('./Result/{}/RL_change_penalty_img_'.format(config['model'])+str(config['RL_epoch'])+'batch_size'+str(config['batch_size']),change_arr)
        np.save('./Result/{}/RL_change_penalty_label_'.format(config['model'])+str(config['RL_epoch'])+'batch_size'+str(config['batch_size']),label)
        np.save('./Result/{}/RL_change_penalty_model_label_'.format(config['model'])+str(config['RL_epoch'])+'batch_size'+str(config['batch_size']),label_o_arr)
        np.save('./Result/{}/RL_change_penalty_model_change_'.format(config['model'])+str(config['RL_epoch'])+'batch_size'+str(config['batch_size']),label_c_arr)

    except Exception as e:
        print(traceback.format_exc())
    return agent.state_dict()


def main():
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    seed = 12
    config['device'] = device

    print("Device name: {}".format(torch.cuda.get_device_name(0)))
    print("Using Device: {}".format(device))
    print("Seed: {}".format(seed))
    seed_all(seed)

    print("Load pretrained CNN model")

    # sweep_id = wandb.sweep(
    #     sweep_config, project="Adversarial_RL", entity="adversarial-rl")
    # # wandb.agent(sweep_id, train)
    # sweep_id = wandb.sweep(sweep_config, project="test", entity="adversarial-rl")
    # policy_dict = wandb.agent(sweep_id, train)
    # policy_dict = agent.state_dict()
    policy_dict=train()
    return policy_dict


sweep_config_defaults = {
            'RL_learning_rate': 0.00001,
            'action_std': 2,
            'decay_rate': 0.99,
        }

wandb.init(project="adversarial", entity="sharpit",config=sweep_config_defaults, reinit=True)
        
run_name = wandb.run.name 
wandb.run.name = 'feedback_'+config['model']+'_label_reward_penalty_'+str(config['RL_epoch'])+'epoch and batch'+str(config['batch_size'])
wandb.run.save()

if __name__ == '__main__':
    img = torch.Tensor(train_data.data).view(-1,1,28,28)/255
    img = img.to(device)
    label = train_data.train_labels.clone()
    for epoch in range(1):
        ##### i is adversarial model 
        image = img.clone()
        # np.save('./Result/original/{}.npy'.format(config['RL_epoch']),image.cpu())
        if config['model'] == 'x_y_linear' or config['model'] == 'heat_linear' :
            
            x_range = torch.linspace(-1, 1, 28)
            y_range = torch.linspace(-1, 1, 28)
            x, y = torch.meshgrid(
            x_range, y_range, indexing="xy")
            x = x.expand([img.shape[0], 1, -1, -1])
            y = y.expand([img.shape[0], 1, -1, -1])
            normalized_coord = torch.cat([x, y], dim=1).to(device)  # [batch_size,2,28,28]
            image = torch.cat([img, normalized_coord], dim=1)

        


        if config['model'] == 'heat_learn' or config['model'] == 'heat_linear':
            cnn.eval()
            ex_cl = ex_class(cnn,img,label)
            heat = ex_cl.heat()
            new_x = torch.cat((image,torch.Tensor(heat.reshape(-1,1,28,28)).to(device)),axis=1)
            new_train_data = train_function.BasicDataset(new_x,label)
            train_loader = torch.utils.data.DataLoader(
            dataset=new_train_data, batch_size=config['batch_size'], shuffle=True
        )   


        
        # main(epoch)

        policy.load_state_dict(main())
        # policy.load_state_dict(torch.load( "./feedback/{}_sweep_".format(config['model'])+str(epoch)+'.pt'))

        cnn.eval()
        policy.eval()

        # img = torch.Tensor(train_data.data).view(-1,1,28,28)/255
        # img = img.to(device)
        # label = train_data.train_labels.clone()



        # label = np.load('./img/y_o.npy')
        # label = torch.LongTensor(label).to(device)

        # out=policy(img, policy.positions)
        if config['model'] =='learnable_cat':
            pos= policy.positions.reshape(1,2,28,28).to(device)
        else:
            pos= policy.positions.reshape(1,1,28,28).to(device)
        



        if config['model'] == 'x_y_linear':
            

            ind = 0
            # print(img.shape)
            adv = ADV(policy,image,label, ind,pos)
            img_x, img_y= adv.dl_3d()
            adv_ex, class_l, class_m, class_c, num = adv.gen_linear(cnn,img_x, img_y)

        if config['model'] == 'learnable_cat':
            

            ind = 0
            # print(img.shape)
            adv = ADV(policy,image,label, ind,pos)
            img_x, img_y= adv.dl_3d()
            adv_ex, class_l, class_m, class_c, num = adv.gen_linear(cnn,img_x, img_y)
            np.save('./Result/{}/heat_x_'.format(config['model'])+str(epoch+1)+'batch_size_'+str(config['batch_size']),img_x)
            np.save('./Result/{}/heat_y_'.format(config['model'])+str(epoch+1)+'batch_size_'+str(config['batch_size']),img_y)


        elif config['model'] == 'learnable_pos' :

            ind = [0,1]
            # pos= policy.positions.reshape(1,1,28,28).to(device)

            adv = ADV(policy,img,label, ind, pos)

            img_x, img_y, pos_x, pos_y = adv.dl_learnable()
            adv_ex, class_l, class_m, class_c, num = adv.gen_learnable(cnn,img_x, img_y, pos_x, pos_y)
            # np.save('./Result/{}/heat_pos_{}_'.format(config['model'],config['RL_epoch'])+str(epoch+1),np.concatenate((pos_x,pos_y),axis=0))
            

        elif config['model'] == 'heat_learn' :

            ind = [0,1]
            # pos= policy.positions.reshape(1,1,28,28).to(device)

            adv = ADV(policy,new_x,label, ind, pos)
            # print(new_x.shape)
            img_x, img_y, pos_x, pos_y = adv.dl_heat_learn()
            adv_ex, class_l, class_m, class_c, num = adv.gen_heat_learn(cnn,img_x, img_y, pos_x, pos_y)
            # np.save('./Result/{}/heat_pos_{}_'.format(config['model'],config['RL_epoch'])+str(epoch+1),np.concatenate((pos_x,pos_y),axis=0))
            
            
        elif config['model'] == 'heat_linear' :


            ind = 0
            

            adv = ADV(policy,new_x,label, ind,pos)
            img_x, img_y= adv.dl_heat_linear()
            adv_ex, class_l, class_m, class_c, num = adv.gen_heat_linear(cnn,img_x, img_y)
        

        elif config['model'] == 'circle':

            ind = 0
            

            adv = ADV(policy,img,label, ind,pos)
            img_x, img_y, pos_x, pos_y = adv.dl_circle()
            # np.save('./Result/{}/heat_pos_{}_'.format(config['model'],config['RL_epoch'])+str(epoch+1),np.concatenate((pos_x,pos_y),axis=0))

        elif config['model'] == 'normal':

            ind = [0,1]
            

            adv = ADV(policy,1-img,label, ind,pos)
            img_x, img_y = adv.dl_normal()
            np.save('./Result/{}/heat_x_'.format(config['model'])+str(epoch+1)+'batch_size'+str(config['batch_size']),img_x)
            np.save('./Result/{}/heat_y_'.format(config['model'])+str(epoch+1)+'batch_size'+str(config['batch_size']),img_y)
        # np.save('./Result/{}/heat_img_{}_'.format(config['model'],config['RL_epoch'])+str(epoch+1),np.concatenate((img_x,img_y),axis=0))
        

        wandb.log({"adv_num" : len(adv_ex), "possible_img_num" : num})
        np.save('./Result/{}/adversarial_'.format(config['model'])+str(epoch+1),adv_ex)
        np.save('./Result/{}/label_'.format(config['model'])+str(epoch+1),class_l)
        np.save('./Result/{}/change_label_'.format(config['model'])+str(epoch+1),class_c)
        np.save('./Result/{}/model_label_'.format(config['model'])+str(epoch+1),class_m)
        np.save('./Result/{}/exam_num_'.format(config['model'])+str(epoch+1),num)



        print('step')
        adv_ex = torch.Tensor(adv_ex).to(device)
        class_l = torch.LongTensor(class_l)
        new_x = torch.cat((img,adv_ex))
        new_y = torch.cat((train_data.train_labels,class_l))
        new_train_data = train_function.BasicDataset(new_x,new_y)
        

        


        new_train_loader = torch.utils.data.DataLoader(
        dataset=new_train_data, batch_size=config['batch_size'], shuffle=True
    )   
        result = train_function.training(cnn, new_train_loader)
        # torch.save(cnn.state_dict(), "./feedback/{}_".format(config['model']+str(i)+'.pt'))


    print('step')
