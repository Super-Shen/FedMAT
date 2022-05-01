import torch
import torch.nn as nn


class cnn_block(nn.Module):
    """↑ fht:
    
    3D conv:
    (128, 1, 5, 3+3+3, 11) -> k(1,3,3),s(1,3,3),p(0,0,2) ->(128, 64, 5, 3, 5)
    (128, 64, 5, 3, 5)     -> k(1,1,3),s(1,1,2),p(0,0,0) ->(128, 64, 5, 3, 2)
    (128, 64, 5, 3, 2)     -> k(1,1,2),s(1,1,2),p(0,0,0) ->(128, 64, 5, 3, 1)

    4D conv:
    (128, 1, 5, 4+4+4, 11) -> k(1,4,3),s(1,4,3),p(0,0,2) ->(128, 64, 5, 3, 5)
    (128, 64, 5, 3, 5)     -> k(1,1,3),s(1,1,2),p(0,0,0) ->(128, 64, 5, 3, 2)
    (128, 64, 5, 3, 2)     -> k(1,1,2),s(1,1,2),p(0,0,0) ->(128, 64, 5, 3, 1)

    merge:
    (128, 1, 5, 6, 64)     -> k(1,6,1),s(1,6,1),p(0,0,0) ->(128, 64, 5, 1, 64)
    (128, 64, 5, 1, 64)    -> k(1,1,8),s(1,1,8),p(0,0,0) ->(128, 64, 5, 1, 8)
    (128, 64, 5, 1, 8)     -> k(1,1,4),s(1,1,2),p(0,0,0) ->(128, 64, 5, 1, 3)
    
    2022-04-26"""
    def __init__(self, input_channels, output_channels, zero_prob):
        

        super(cnn_block, self).__init__()

        # 2022-04-26 fht: 3D sensors
        self.d3_conv1 = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=(1, 3, 3), stride=(1, 3, 3), padding=(0, 0, 2)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
            nn.Dropout(p=zero_prob),
        )
        self.d3_conv2 = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, kernel_size=(1, 1, 3), stride= (1, 1, 2), padding=(0, 0, 0)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
        )
        self.d3_conv3 = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, kernel_size=(1, 1, 2), stride= (1, 1, 2), padding=(0, 0, 0)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
        )

        # 2022-04-26 fht:  4D sensors
        self.d4_conv1 = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=(1, 4, 3), stride=(1, 4, 3), padding=(0, 0, 2)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
            nn.Dropout(p=zero_prob),
        )
        self.d4_conv2 = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, kernel_size=(1, 1, 3), stride= (1, 1, 2), padding=(0, 0, 0)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
        )
        self.d4_conv3 = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, kernel_size=(1, 1, 2), stride= (1, 1, 2), padding=(0, 0, 0)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
        )

        # 2022-04-26 fht: merge
        self.merge_conv1 = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=(1, 6, 1), stride= (1, 6, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
        )
        self.merge_conv2 = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, kernel_size=(1, 1, 8), stride= (1, 1, 8), padding=(0, 0, 0)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
        )
        self.merge_conv3 = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, kernel_size=(1, 1, 4), stride= (1, 1, 2), padding=(0, 0, 0)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        
        d3_conv1 = self.d3_conv1(x1)
        d3_conv2 = self.d3_conv2(d3_conv1)
        d3_conv_rs = self.d3_conv3(d3_conv2)
        
        d4_conv1 = self.d4_conv1(x2)
        d4_conv2 = self.d4_conv2(d4_conv1)
        d4_conv_rs = self.d4_conv3(d4_conv2)
        
        
        d3_final = torch.reshape(d3_conv_rs, [d3_conv_rs.shape[0], 1, d3_conv_rs.shape[2], d3_conv_rs.shape[3], -1]) # 2022-04-29 fht: [BN, 1, 5, 3, 64]
        d4_final = torch.reshape(d4_conv_rs, [d4_conv_rs.shape[0], 1, d4_conv_rs.shape[2], d4_conv_rs.shape[3], -1]) # 2022-04-29 fht: [BN, 1, 5, 3, 64]
        
        merge_in = torch.cat([d3_final, d4_final], 3)  # 2022-04-29 fht: [BN, 1, 5, 6, 64]

        merge_out1 = self.merge_conv1(merge_in) # 2022-01-07 [BN, 64, 10, 1, 64]
        merge_out2 = self.merge_conv2(merge_out1) # 2022-01-07 [BN, 64, 10, 1, 8]
        merge_out3 = self.merge_conv3(merge_out2) # 2022-01-07 [BN, 64, 10, 1, 3]

        return [x1, d3_conv1, d3_conv2, x2, d4_conv1, d4_conv2, merge_in, merge_out1, merge_out2, merge_out3]

class soft_Attn_Merge(nn.Module):
    def __init__(self, input_channels, output_channels, zero_prob):


        super(soft_Attn_Merge, self).__init__()
        
        self.mask_d3 = [
            self.gen_mask_conv(input_channels=1, output_channels=1),
            self.gen_mask_conv(input_channels=64, output_channels=64),
            self.gen_mask_conv(input_channels=64, output_channels=64),
            # self.gen_mask_conv(input_channels=64, output_channels=64),
        ]
        self.mask_d4 = [
            self.gen_mask_conv(input_channels=1, output_channels=1),
            self.gen_mask_conv(input_channels=64, output_channels=64),
            self.gen_mask_conv(input_channels=64, output_channels=64),
            # self.gen_mask_conv(input_channels=64, output_channels=64),
        ]
        self.mask_merge = [
            self.gen_mask_conv(input_channels=1, output_channels=1),
            self.gen_mask_conv(input_channels=64, output_channels=64),
            self.gen_mask_conv(input_channels=64, output_channels=64),
            self.gen_mask_conv(input_channels=64, output_channels=64)
        ]
        self.mask_d3 = nn.ModuleList(self.mask_d3)
        self.mask_d4 = nn.ModuleList(self.mask_d4)
        self.mask_merge = nn.ModuleList(self.mask_merge)
        
        # 2022-04-26 fht: 3D sensors
        self.d3_conv1 = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=(1, 3, 3), stride=(1, 3, 3), padding=(0, 0, 2)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
            nn.Dropout(p=zero_prob),  # 2022-04-26 fht: 只对原始输入使用Dropout
        )        
        self.d3_conv2 = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, kernel_size=(1, 1, 3), stride= (1, 1, 2), padding=(0, 0, 0)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
        )
        self.d3_conv3 = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, kernel_size=(1, 1, 2), stride= (1, 1, 2), padding=(0, 0, 0)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
        )

        # 2022-04-26 fht:  4D sensors  (6, 12, 26)->(6, 13)
        self.d4_conv1 = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=(1, 4, 3), stride=(1, 4, 3), padding=(0, 0, 2)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
            nn.Dropout(p=zero_prob),
        )
        self.d4_conv2 = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, kernel_size=(1, 1, 3), stride= (1, 1, 2), padding=(0, 0, 0)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
        )
        self.d4_conv3 = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, kernel_size=(1, 1, 2), stride= (1, 1, 2), padding=(0, 0, 0)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
        )

        # 2022-04-26 fht: merge
        self.merge_conv1 = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=(1, 6, 1), stride= (1, 6, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
        )
        self.merge_conv2 = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, kernel_size=(1, 1, 8), stride= (1, 1, 8), padding=(0, 0, 0)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
        )
        self.merge_conv3 = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, kernel_size=(1, 1, 4), stride= (1, 1, 2), padding=(0, 0, 0)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
        )

    def gen_mask_conv(self, input_channels, output_channels):
        attn_mask = nn.Sequential(
            nn.Conv3d(input_channels, input_channels, kernel_size=(1, 2, 1), stride=(1, 2, 1)),
            nn.BatchNorm3d(input_channels),
            nn.ReLU(),  # y in [0,)
            
            nn.Conv3d(input_channels, output_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(input_channels),
            nn.Sigmoid()  # y(mask) in [0,1]
        )
        return attn_mask
    
    def forward(self, shared_rs):
        
        # 3D sensor
        shared_3d = [shared_rs[0], shared_rs[1], shared_rs[2]]
        d3_in = self.my_cat(shared_3d[0], shared_3d[0])  # 2022-04-30 fht: [BN, 64, 5, 18, 11]
        d3_mask = self.mask_d3[0](d3_in)  # 2022-04-30 fht: [BN, 64, 5, 9, 11]
        d3 = d3_mask*shared_3d[0]  # 2022-04-30 fht: [BN, 64, 5, 9, 11]
        d3_conv1_attn = self.d3_conv1(d3)  # 2022-04-30 fht: [BN, 64, 5, 3, 5]
        
        d3_conv1 = shared_3d[1]  # 2022-01-07 [BN, 64, 10, 4, 8]
        d3_conv1_cat = self.my_cat(d3_conv1, d3_conv1_attn)
        d3_conv1_mask = self.mask_d3[1](d3_conv1_cat)  # 2022-01-07 [BN, 64, 10, 4, 8]
        d3_conv1 = d3_conv1_mask*d3_conv1
        d3_conv2_attn = self.d3_conv2(d3_conv1)
        
        d3_conv2 = shared_3d[2]
        d3_conv2_cat = self.my_cat(d3_conv2, d3_conv2_attn)
        d3_conv2_mask = self.mask_d3[1](d3_conv2_cat)
        d3_conv2 = d3_conv2_mask*d3_conv2
        d3_conv3_attn = self.d3_conv3(d3_conv2)
        

        # 4D sensor
        shared_4d = [shared_rs[3], shared_rs[4], shared_rs[5]]
        d4_in = self.my_cat(shared_4d[0], shared_4d[0])
        d4_mask = self.mask_d4[0](d4_in)
        d4 = d4_mask*shared_4d[0]
        d4_conv1_attn = self.d4_conv1(d4)
        

        d4_conv1 = shared_4d[1]
        d4_conv1_cat = self.my_cat(d4_conv1, d4_conv1_attn)
        d4_conv1_mask = self.mask_d4[1](d4_conv1_cat)  # 2022-01-07 [BN, 64, 10, 4, 8]
        d4_conv1 = d4_conv1_mask*d4_conv1
        d4_conv2_attn = self.d4_conv2(d4_conv1)
        

        d4_conv2 = shared_4d[2]
        d4_conv2_cat = self.my_cat(d4_conv2, d4_conv2_attn)
        d4_conv2_mask = self.mask_d4[1](d4_conv2_cat)
        d4_conv2 = d4_conv2_mask*d4_conv2
        d4_conv3_attn = self.d4_conv3(d4_conv2)
        
        
        
        d3_final = torch.reshape(d3_conv3_attn, [d3_conv3_attn.shape[0], 1, d3_conv3_attn.shape[2], d3_conv3_attn.shape[3],  -1])
        d4_final = torch.reshape(d4_conv3_attn, [d4_conv3_attn.shape[0], 1, d4_conv3_attn.shape[2], d4_conv3_attn.shape[3], -1])
        merge_in_attn = torch.cat([d3_final, d4_final], 3)
        

        shared_merge = [ shared_rs[6], shared_rs[7],shared_rs[8], shared_rs[9]]
        merge_in = shared_merge[0]
        merge_in_cat = self.my_cat(merge_in, merge_in_attn)
        merge_in_mask = self.mask_merge[0](merge_in_cat)
        merge_in = merge_in_mask * merge_in
        merge_conv1_attn = self.merge_conv1(merge_in)
        

        merge_out1 = shared_merge[1]
        merge_out1_cat = self.my_cat(merge_out1, merge_conv1_attn)
        merge_out1_mask = self.mask_merge[1](merge_out1_cat)
        merge_out1 = merge_out1_mask * merge_out1
        merge_conv2_attn = self.merge_conv2(merge_out1)
        

        merge_out2 = shared_merge[2]
        merge_out2_cat = self.my_cat(merge_out2, merge_conv2_attn)
        merge_out2_mask = self.mask_merge[2](merge_out2_cat)
        merge_out2 = merge_out2_mask * merge_out2
        merge_conv3_attn = self.merge_conv3(merge_out2)
        

        merge_out3 = shared_merge[3]
        merge_final_cat = self.my_cat(merge_out3, merge_conv3_attn)
        merge_final_mask = self.mask_merge[3](merge_final_cat)
        merge_final = merge_final_mask * merge_out3
        
        merge_final = torch.reshape(merge_final, [merge_final.shape[0], merge_final.shape[2], -1]) # 2022-04-26 fht: [BN, 6, 3*64]
        
        return merge_final
    
    def get_mask(self):
        merge_mask = self.mask_merge.state_dict()
        return merge_mask
    
    def my_cat(self, shared_tensor, attn_tensor):

        f_shape = [shape for shape in attn_tensor.shape]
        f_shape[-2] *= 2

        cat_1 = torch.cat([shared_tensor, attn_tensor], dim=-1)
        cat_2 = torch.reshape(cat_1, f_shape)

        return cat_2

class rnn_block(nn.Module):
    """↑
    
    [seq_len, BN, input_size] -> [seq_len, BN, num_directions*hidden_size]
    
    2021-10-01"""
    def __init__(self, hidden_size, num_layers, bidirectional=False):
        super(rnn_block, self).__init__()
        """↑
        
        Args:
            feature_len: input.shape[-1]
            hidden_size: output.shape[-1]
        

        
        2021-10-01"""
        self.input_size=3*64
        self.hidden_size=hidden_size
        self.num_layers=num_layers



        # self.lstm = nn.LSTM(
        #     input_size=self.input_size,
        #     hidden_size=self.hidden_size,
        #     num_layers=num_layers,
        #     bidirectional=bidirectional
        #     )
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional
            )
    
    def forward(self, x):
        rnn_out, _ = self.gru(x)
        rnn_out = torch.mean(rnn_out, dim=1)
        return rnn_out

class fully_connected(nn.Module):
    def __init__(self, label_num):
        super(fully_connected, self).__init__()
        self.fc = nn.Linear(100, label_num)

    def forward(self, x):
        logits = self.fc(x)
        return logits

class server(nn.Module):
    def __init__(self, out_channel, dropout_prob, rnn_hidden_size):
        super(server, self).__init__()
        
        self.cnns = cnn_block(input_channels=1, output_channels=out_channel, zero_prob=dropout_prob)
        
        self.rnn = rnn_block(hidden_size=rnn_hidden_size, num_layers=2)
        
    def forward(self, x, with_rnn=True):
        # 2022-04-26 fht: [BN, 1, 6, 4+4+4+3+3+3, 26]
        x2, x1 = x.split([12,9], dim=3)

        shared_rs = self.cnns(x1, x2)

        if with_rnn:
            merge_out = shared_rs[-1]
            merge_out = torch.reshape(merge_out,[merge_out.shape[0], merge_out.shape[2], -1])  # 2022-04-28 fht: [BN, 6, 3*64]
            rnn_out = self.rnn(merge_out)
            return rnn_out
        else:
            return shared_rs

    
    
    def get_rnn_weights(self):
        return self.rnn.state_dict()
    
    def get_server_weights(self):
        return [
            self.cnns.d3_conv1.state_dict(), self.cnns.d3_conv2.state_dict(), self.cnns.d3_conv3.state_dict(),
            self.cnns.d4_conv1.state_dict(), self.cnns.d4_conv2.state_dict(), self.cnns.d4_conv3.state_dict(),
            self.cnns.merge_conv1.state_dict(), self.cnns.merge_conv2.state_dict(), self.cnns.merge_conv3.state_dict(),
            self.rnn.state_dict()
        ]
    

class client_sp(nn.Module):
    def __init__(self, out_channel, dropout_prob, rnn_hidden_size, label_num):
        super(client_sp, self).__init__()
        
        self.soft_attn = soft_Attn_Merge(input_channels=1, output_channels=out_channel, zero_prob=dropout_prob)
        
        self.rnn = rnn_block(hidden_size=rnn_hidden_size, num_layers=2)

        self.label_num = label_num
        self.fc = fully_connected(label_num=self.label_num)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, merge_shared_rs):
        merge_out = self.soft_attn(merge_shared_rs)
        # merge_out: [BN, 6, 3*64]
        
        rnn_shared_out = self.rnn(merge_out)
        # rnn_out: [BN, 100]
        
        logits = self.fc(rnn_shared_out)
        # logits: [BN, 7]
        
        logits = self.softmax(logits)
        # logits: [BN, 7]
        
        return logits
    
    def get_attn_w(self):
        all_attn_w = self.soft_attn.get_mask()
        return all_attn_w
    
    def apply_server_rnn_weights(self, rnn_w):
        self.rnn.load_state_dict(rnn_w)
        return
    
    def get_rnn_weights(self):
        return self.rnn.state_dict()
    
    def set_attn_weights(self, server_cnn_weights):
        self.soft_attn.d3_conv1.load_state_dict(server_cnn_weights[0])
        self.soft_attn.d3_conv2.load_state_dict(server_cnn_weights[1])
        self.soft_attn.d3_conv3.load_state_dict(server_cnn_weights[2])
        self.soft_attn.d4_conv1.load_state_dict(server_cnn_weights[3])
        self.soft_attn.d4_conv2.load_state_dict(server_cnn_weights[4])
        self.soft_attn.d4_conv3.load_state_dict(server_cnn_weights[5])
        self.soft_attn.merge_conv1.load_state_dict(server_cnn_weights[6])
        self.soft_attn.merge_conv2.load_state_dict(server_cnn_weights[7])
        self.soft_attn.merge_conv3.load_state_dict(server_cnn_weights[8])
        
def get_server(out_channel, dropout_prob, rnn_hidden_size):
    return server(out_channel, dropout_prob, rnn_hidden_size)

def get_client_sp(output_channel, dropout_prob, rnn_hidden_size, label_num):
    return client_sp(output_channel, dropout_prob, rnn_hidden_size, label_num)



def main():
    import pandas as pd
    import numpy as np
    all_trans_dict = pd.read_pickle("/myData/iLog/iLog数据处理代码/dataset_10_2/transfer_dict_info/whatdoing.pickle")
    all_act_num = {}
    for user in all_trans_dict.index:
        all_act_num[user] = user_act_num = np.sum(all_trans_dict.loc[user] != -1)
    user_id = "a1be2dfcd1a9133fe88a1349039616dce39e3326"
    label_num = all_act_num[user_id]

    # x = torch.randn(128, 1, 10, 8, 16)
    x = torch.randn(128, 1, 5, 21, 11)
    # x1, x2, x3 = x.split([6,12,3], dim=3)
    # x1 = torch.cat([x1, x3], dim=3)
    # print(x1.shape)
    # print(x2.shape)
    print(x.shape)
    print("")

    server_model = get_server(out_channel=64, dropout_prob=0.2, rnn_hidden_size=100)
    a_client_sp = get_client_sp(output_channel=64, dropout_prob=0.2, rnn_hidden_size=100, label_num=label_num)
    
    shared_rs = server_model(x)
    print(shared_rs.shape)
    print("")
    
    shared_rs = server_model(x, with_rnn=False)
    for rs in shared_rs:
        print(rs.shape)
    
    print("")
    logits = a_client_sp(shared_rs)
    print(logits.shape)
    
    # model = MyEnsemble(server_model, a_client_sp)
    # # print(model)
    # # model.set_user_sp()
    # logits = model(x)


def state_dict_test():
    server_model = get_server(out_channel=64, dropout_prob=0.2, rnn_hidden_size=100)
    a_client_sp = get_client_sp(output_channel=64, dropout_prob=0.2, rnn_hidden_size=100, label_num=23)
    print()

if __name__ == "__main__":
    # state_dict_test()
    main()

