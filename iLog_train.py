import os
import torch
import random
import torch.nn as nn
# import utils_hhar as utils
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import copy
import iLog_models as base
import numpy as np
import pandas as pd
from sklearn import metrics
import iLog_data_loader as myLoader
import datetime
import wandb

total_user = [
    "a1be2dfcd1a9133fe88a1349039616dce39e3326", "a32dcba8e90ceb8b61770ad8a71aa4245ea7f0d6",
    "a7b26dcf785d2479dcb9177db235bd9cf7ebb4fd", "a88c1234a960cbc0dac8bf8aec7e5e9ce2c13fe1",
    "ba349bbf09e99793aa1ef88448909ec5ba3a392c", "bbb94fade1bc0dbb72429f0b959e9882869a86c7",
    "c4040f1d90cf8f933dbda3674e45fcadfa94c08d", "cb54af7747cdbfd18cdf848d376e5f8475951d38",
    "e3daf75e9bd1d643749f1fd8ad42815088847fbd", "e972bd8264d6604a8c0be717f40a23d8ce755db9",
]


# 获取用户本地活动数量
# 使用方法：
# user_id = "a1be2dfcd1a9133fe88a1349039616dce39e3326"
# label_num = all_act_num[user_id]
all_trans_dict = pd.read_pickle("/myData/iLog/iLog数据处理代码/dataset_10_2/transfer_dict_info/whatdoing.pickle")
user_act_num = {}
for user in all_trans_dict.index:
    user_act_num[user] = np.sum(all_trans_dict.loc[user] != -1)

user_act_num["server"] = 23



config = {
        "project_name": "FEDMAT_ilog_w30",
        
        "sensor_num": 6,
        "out_channel": 64,
        "dropout_prob": 0.2, 
        "rnn_hidden_size": 100, 
        "label_kind": "whatdoing",
        "user_act_num": user_act_num,
        
        "user_list": total_user,
        "train_users": total_user[:9],  # list
        'choose_num': 5,
        "test_users": total_user[-1],  # str
        "batch_size": 64,
        "ds_pth": "/myData/iLog/iLog数据处理代码/dataset_10_2/byuser",
        "trans_dict_pth": "/myData/iLog/iLog数据处理代码/dataset_10_2/transfer_dict_info/whatdoing.pickle",
        "device": 'cuda:0' if torch.cuda.is_available() else 'cpu',
        
        "weights_folder_name": "weight_train19_test-1",
        
        "run_cmd": "nohup python -u iLog_train.py >20220501_1528_train.log 2>&1",
        
        "lr": 1e-3,
        "weight_decay": 1e-8,
        "adam_betas":[0.9, 0.98],
        "schedular_step_size": 2,  
        "schedular_gamma": 0.5,
        "seed": 20220426,
        
        "total_rounds": 900,
        "test_rounds_start":50,
        "local_epoch": 5,
        "update_w": 0.5,  # lambda
        "local_tuning": [2, 4]  # 实际是*local_epoch
    }

wandb.init(project=config['project_name'], tags=["FedMAT"], config=config)


class Mypairwiseloss(torch.nn.Module):
    def __init__(self):
        super(Mypairwiseloss, self).__init__()

    def forward(self, embed, target, chunck_size, Expand=10):


        Expand = np.float64(Expand * 1.0)
        # print(embed.shape)
        embedded_split = torch.split(embed, chunck_size, dim=0)  # 2021-09-26 fht: <e_i, e_j>
        targets_split = torch.split(target, chunck_size, dim=0)  # 2021-09-26 fht: <a_i, a_j>

        Phi = cosine_similarity(embedded_split[0], embedded_split[1])*Expand  # 2021-09-27 fht: [128]

        sig_Phi = torch.sigmoid(Phi)  # 2021-09-27 fht: [128]
        log_sig_Phi = torch.log(sig_Phi)
        _log_sig_Phi = torch.log(torch.sub(1, sig_Phi)) # 2021-09-27 


        mask = torch.sum(torch.mul(targets_split[0], targets_split[1]), dim=1)  # 2021-09-27 fht: [128]
        _mask = torch.sub(1, mask)

        pairwise_loss = -1 * (torch.add(torch.mul(mask, log_sig_Phi), torch.mul(_mask, _log_sig_Phi))).mean()
        return pairwise_loss

cross_entropy = nn.CrossEntropyLoss()
cosine_similarity = nn.CosineSimilarity()
pairwiseloss = Mypairwiseloss()
# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'



class MyEnsemble(nn.Module):
    def __init__(self, server, client_sp):
        super(MyEnsemble, self).__init__()
        self.server = server
        self.client_sp = client_sp

    def forward(self, x):
        shared_rs = self.server(x, with_rnn=False)
        logits = self.client_sp(shared_rs)
        return logits



        
class meta_train(object):
    def __init__(self, user_id="server"):
        super(meta_train, self).__init__()
        self.lr = config["lr"]
        # self.num_class = config["label_num"]  
        self.device = config['device']
        print("training setup on device:", self.device)
        
        self.embedded_model = base.get_server(config["out_channel"], config["dropout_prob"], config["rnn_hidden_size"])
        self.embedded_model = self.embedded_model.to(self.device)
        
        self.client_model = base.get_client_sp(output_channel=config["out_channel"], rnn_hidden_size=config["rnn_hidden_size"], dropout_prob=config["dropout_prob"],label_num=config["user_act_num"][user_id])
        self.client_model = self.client_model.to(self.device)
        self.user_model = MyEnsemble(self.embedded_model, self.client_model)
        self.training_op = {}
        self.training_op["optimizer_embedded"] = optim.Adam(self.embedded_model.parameters(), lr=self.lr, weight_decay=config["weight_decay"], betas=config["adam_betas"])  # 2021-10-02 fht: 0: 1e-8, [0.9, 0.98]
        self.training_op["optimizer_user"] = optim.Adam(self.user_model.parameters(), lr=self.lr, weight_decay=config["weight_decay"], betas=config["adam_betas"])
        self.training_op["loss_func_embedded"] = pairwiseloss
        self.training_op["loss_func_user"] = cross_entropy
        self.training_op["scheduler_embedded"] = StepLR(self.training_op["optimizer_embedded"], step_size=2, gamma=0.85)
        self.training_op["scheduler_user"] = StepLR(self.training_op["optimizer_user"], step_size=2, gamma=0.85)

        self.user_name = None
        self.ds_pth = config["ds_pth"]
        self.training_op["train_file"] = None
        self.training_op["test_file"] = None
        self.training_op["adapt_file"] = None

        self.training_op["train_loader"] = None
        self.training_op["test_loader"] = None
        self.training_op["adapt_loader"] = None

    def reset_optimizer_schedular(self):
        

        self.training_op["optimizer_embedded"] = optim.Adam(self.embedded_model.parameters(), lr=self.lr, weight_decay=config["weight_decay"])
        self.training_op["optimizer_user"] = optim.Adam(self.user_model.parameters(), lr=self.lr, weight_decay=config["weight_decay"])

        self.training_op["scheduler_embedded"] = StepLR(self.training_op["optimizer_embedded"], step_size=config["schedular_step_size"], gamma=config["schedular_gamma"])
        self.training_op["scheduler_user"] = StepLR(self.training_op["optimizer_user"], step_size=config["schedular_step_size"], gamma=config["schedular_gamma"])

    #OK
    def build_data_loader(self, user_name):
        
        self.user_name = user_name
        user_ds_pth = os.path.join(self.ds_pth, self.user_name)
        
        self.training_op["train_loader"] = myLoader.get_dataloader(ds_pth=user_ds_pth, mode="train", batch=config["batch_size"]*2)

        self.training_op["test_loader"] = myLoader.get_dataloader(ds_pth=user_ds_pth, mode="test", batch=config["batch_size"])

    # OK
    def get_embedded_weights(self): 
        return self.embedded_model.state_dict()
    
    # OK
    def set_new_weights(self, server_weights_dict):  # 2021-10-01 
        self.embedded_model.load_state_dict(server_weights_dict)
    
    # OK
    def train_embedded(self, num_epoch):

        self.reset_optimizer_schedular()

        for epoch in range(num_epoch):
            self.embedded_model.train()
            loss = 0
            for _, (inputs, targets, _) in enumerate(self.training_op["train_loader"]):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.training_op["optimizer_embedded"].zero_grad()
                if "cuda" in self.device:
                    embeddings = self.embedded_model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
                    loss = self.training_op["loss_func_embedded"](embeddings, targets, config["batch_size"])
                else:
                    embeddings = self.embedded_model(inputs.unsqueeze(1).type(torch.FloatTensor))
                    loss = self.training_op["loss_func_embedded"](embeddings, targets, config["batch_size"])
                loss.backward()
                self.training_op["optimizer_embedded"].step()
            self.training_op["scheduler_embedded"].step()

    # OK
    def adapt_user(self, num_epoch, loader_type="adapt"):

        if loader_type == "train":
            adapt_loader = self.training_op["train_loader"]
        elif loader_type == "adapt":
            adapt_loader = self.training_op["train_loader"]
        
        self.reset_optimizer_schedular()

        # 2021-09-25  embedded network，pairwise loss
        self.embedded_model.train()
        loss = 0
        for epoch in range(num_epoch):
            for _, (inputs, targets, _) in enumerate(adapt_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.training_op["optimizer_embedded"].zero_grad()
                if "cuda" in self.device:
                    embeddings = self.embedded_model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
                    loss = self.training_op["loss_func_embedded"](embeddings, targets, config["batch_size"])
                else:
                    embeddings = self.embedded_model(inputs.unsqueeze(1).type(torch.FloatTensor))
                    loss = self.training_op["loss_func_embedded"](embeddings, targets, config["batch_size"])
                loss.backward()
                self.training_op["optimizer_embedded"].step()
            self.training_op["scheduler_embedded"].step()
        

        # softAttn conv module shared embedded networks' weights
        # then fix embedded network 
        # only train the client_model [SoftAttn+rnn+last layer] with CE Loss
        
        fine_tunned_server_weights = self.embedded_model.get_server_weights()
        self.client_model.set_attn_weights(fine_tunned_server_weights)
        self.client_model.apply_server_rnn_weights(fine_tunned_server_weights[-1])
        self.embedded_model.eval()  # eval()模式，不会使用BN和Dropout，正常计算梯度，但是不参与反向传播
        self.client_model.train()  # train()模式，使用BN和Dropout，正常计算梯度，参与反向传播
        
        for epoch in range(num_epoch):
            for _, (inputs, _, targets_t) in enumerate(adapt_loader):
                inputs, targets_t = inputs.to(self.device), targets_t.to(self.device).to(self.device)
                self.training_op["optimizer_user"].zero_grad()
                if "cuda" in self.device:
                    logits_t = self.user_model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
                    # index_targets = torch.argmax(targets, dim=1)
                    loss = self.training_op["loss_func_user"](logits_t, targets_t.type(torch.cuda.LongTensor))
                else:
                    logits_t = self.user_model(inputs.unsqueeze(1).type(torch.FloatTensor))
                    loss = self.training_op["loss_func_user"](logits_t, targets_t.type(torch.LongTensor))
                
                loss.backward()  # 这里由于模块设置，只更新client部分的参数
                self.training_op["optimizer_user"].step()
            self.training_op["scheduler_user"].step()

    # OK
    def test_user(self):
        self.user_model.eval()
        all_targets = []  
        all_predict = []
        for _, (inputs, _, targets_t) in enumerate(self.training_op["test_loader"]):
            inputs = inputs.to(self.device)
            if "cuda" in self.device:
                logits_t = self.user_model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
            else:
                logits_t = self.user_model(inputs.unsqueeze(1).type(torch.FloatTensor))
            predicted = torch.argmax(logits_t, dim=1)  # 2021-09-26 fht: [BN, 7] -> [BN]
            
            all_targets += targets_t.cpu()  # 2021-09-26 fht: [BN]
            all_predict += predicted.cpu()
        
        test_acc = metrics.accuracy_score(all_targets, all_predict)
        test_f1 = np.round(metrics.f1_score(all_targets, all_predict, average='macro'), 4)
        tmp_confusion_metric = metrics.confusion_matrix(y_true=all_targets, y_pred=all_predict)
        return  test_acc, test_f1, len(all_targets), tmp_confusion_metric

    # OK
    def get_confusion_metric(self):
        all_targets = []  
        all_predict = []
        for _, (inputs, targets, targets_t) in enumerate(self.training_op["test_loader"]):
            inputs = inputs.to(self.device)
            if "cuda" in self.device:
                logits_t = self.user_model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
            else:
                logits_t = self.user_model(inputs.unsqueeze(1).type(torch.FloatTensor))
            predicted = torch.argmax(logits_t, dim=1)  # 2021-09-26 fht: [BN, 7] -> [BN]
            all_targets += targets_t.cpu()  # 2021-09-26 fht: [BN]
            all_predict += predicted.cpu()
        
        tmp_confusion_metric = metrics.confusion_matrix(y_true=all_targets, y_pred=all_predict)
        return  tmp_confusion_metric

    # OK
    def get_attn_w(self):
        attn_w = self.client_model.get_attn_w()
        return self.user_name, attn_w
    
    # OK
    def get_model_weights(self):
        return self.user_model.state_dict()


# OK
def get_new_Theta_c(new_weights_list, server_weights, _lambda):

    theta = copy.deepcopy(new_weights_list[0])
    theta_c = {}
    for k in server_weights.keys():

        for idx in range(1, len(new_weights_list)):
            theta[k] += new_weights_list[idx][k]

        theta[k] = torch.div(theta[k], len(new_weights_list))
        theta_c[k] = server_weights[k] + torch.mul(theta[k].sub(server_weights[k]), _lambda)
    return theta_c


def train_setup():
    
    max_test_acc = -1
    chosen_users = config["train_users"]
    test_users = config["test_users"]
    
    train_client_models = []
    for idx in range(len(chosen_users)):
        print("setup", idx+1,"/", len(chosen_users), "train_user:", chosen_users[idx])
        train_client_models.append(meta_train(chosen_users[idx]))
        train_client_models[-1].build_data_loader(user_name=chosen_users[idx])
    
    # test_client_models = []
    # for idx in range(len(test_users)):
    #     print("setup", idx+1,"/", len(test_users), "test_user:", test_users[idx])
    #     test_client_models.append(meta_train(test_users[idx]))
    #     test_client_models[-1].build_data_loader(user_name=test_users[idx])

    
    
    server_model = meta_train()  
    """
    
    
    ↓ 2022-01-07"""
    total_rounds = config['total_rounds']  # 100
    local_epoch = config['local_epoch']  # 5
    update_w = config['update_w']  # 0.1
    for round in range(total_rounds):
        chosen_client = np.random.choice(len(train_client_models), config['choose_num'], replace=False)
        updated_weights = []  
        server_weights = server_model.get_embedded_weights()
        for user_idx in chosen_client:
            print(str(round)+"/"+str(total_rounds)+" rounds         -- Start local training on user:  ", chosen_users[user_idx])
            train_client_models[user_idx].set_new_weights(server_weights)
            train_client_models[user_idx].train_embedded(num_epoch=local_epoch)
            updated_weights.append(train_client_models[user_idx].get_embedded_weights())
        

        print("\n"+str(round)+"/"+str(total_rounds)+" rounds     # update global meta model ========\n")
        server_model.set_new_weights(
            get_new_Theta_c(new_weights_list=updated_weights, server_weights=server_model.get_embedded_weights(), _lambda=update_w)
        )
         
        if round >= config["test_rounds_start"] and round % 2 == 0:
        # if round % 2 == 0:
            user_cms = {}
            adapt_start = datetime.datetime.now()
            local_tuning = config["local_tuning"]
            server_weights = server_model.get_embedded_weights()

            # 测试训练用户效果
            for local_tune in local_tuning:
                adapt_acc = []
                adapt_f1 = []
                label_weights = []
                for user_idx in range(len(train_client_models)):
                    print(str(round)+"/"+str(total_rounds)+" rounds         -- Start local adapting on train_user: ", chosen_users[user_idx])
                    train_client_models[user_idx].set_new_weights(server_weights)
                    train_client_models[user_idx].adapt_user(num_epoch=local_tune)
                    acc, f1, test_num, user_cm = train_client_models[user_idx].test_user()
                    adapt_acc.append(acc)
                    adapt_f1.append(f1)
                    label_weights.append(test_num)
                    user_cms["train_user_"+str(user_idx)+"_tune_"+str(local_tune)] = user_cm
                
                avg_acc = np.average(adapt_acc, weights=label_weights)
                avg_maf1 = np.average(adapt_f1, weights=label_weights)
                print("avg_maf1:", avg_maf1, "avg_acc:", avg_acc, "\n")
                wandb.log(
                    {
                        "train_tune_"+str(local_tune)+"_avgMaF1": avg_maf1,
                        "train_tune_"+str(local_tune)+"_avgAcc": avg_acc,
                    },
                    step=round
                )
            
            # 测试测试用户效果
            for local_tune in local_tuning:
                adapt_acc = []
                adapt_f1 = []
                label_weights = []
                
                # set up test_model
                print("setup 1/1 test_user:", test_users)
                test_client_models = meta_train(test_users)
                test_client_models.build_data_loader(user_name=test_users)

                # adapt and test
                print(str(round)+"/"+str(total_rounds)+" rounds         -- Start local adapting on test_user: ", test_users)
                test_client_models.set_new_weights(server_weights)
                test_client_models.adapt_user(num_epoch=local_tune)
                acc, maf1, test_num, user_cm = test_client_models.test_user()
                user_cms["test_user_"+str(user_idx)+"_tune_"+str(local_tune)] = user_cm
                
                print("test_maf1:", maf1, "test_acc:", acc, "\n")
                wandb.log(
                    {
                        "test_tune_"+str(local_tune)+"_avgMaF1": maf1,
                        "test_tune_"+str(local_tune)+"_avgAcc": acc,
                    },
                    step=round
                )
            
            if acc > max_test_acc:
                max_test_acc = acc
                
                w_folder = config["weights_folder_name"]
                if not os.path.exists(w_folder):
                    os.mkdir(w_folder)
                file_name = os.path.join(w_folder, "server_weights.pt")  # shared_weights: cnn + merge + rnn
                torch.save(server_model.get_embedded_weights(), f=file_name)
                
                np.save("user_cms.npy", user_cms)

                for client_model in train_client_models:
                    user_name, mask = client_model.get_attn_w()
                    file_name = os.path.join(w_folder, "merge_4_mask_"+user_name+".pt")
                    torch.save(mask, f=file_name)
                    
                    client_w = client_model.get_model_weights()
                    file_name = os.path.join(w_folder, "client_"+user_name+".pt")
                    torch.save(client_w, f=file_name)
                
            adapt_cost = datetime.datetime.now()-adapt_start
            print("### adapt_cost:", adapt_cost)
            print("### total time_remain:", adapt_cost*(total_rounds-round))
    return max_test_acc

if __name__ == "__main__":
    train_setup()
    
    # a = datetime.datetime.now()
    # b = datetime.datetime.now()-a
    # c = b*50
    # print(a, b, c)