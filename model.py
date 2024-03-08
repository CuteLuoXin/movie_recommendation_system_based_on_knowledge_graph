import torch.nn as nn
from utils import *
from torch.nn import Module
import scipy.sparse as sp


class GCN_Layer(Module):
    def __init__(self, inF, outF):
        super(GCN_Layer, self).__init__()
        self.W1 = torch.nn.Linear( in_features = inF, out_features = outF)
        self.W2 = torch.nn.Linear( in_features = inF, out_features = outF)

    def forward(self, graph, selfLoop, features):
        part1 = self.W1(torch.sparse.mm(graph + selfLoop, features))
        part2 = self.W2(torch.mul(torch.sparse.mm(graph, features), features))
        return nn.LeakyReLU()(part1 + part2)


class GCN(Module):
    def __init__(self, args, user_feature, item_feature, rating):
        super(GCN, self).__init__()
        self.args = args
        self.device = args.device
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.rating = rating
        self.num_user = rating['user_id'].max() + 1
        self.num_item = rating['item_id'].max() + 1
        # user embedding
        self.user_id_embedding = nn.Embedding(user_feature['id'].max() + 1, 32)
        self.user_age_embedding = nn.Embedding(user_feature['age'].max() + 1, 4)
        self.user_gender_embedding = nn.Embedding(user_feature['gender'].max() + 1, 2)
        self.user_occupation_embedding = nn.Embedding(user_feature['occupation'].max() + 1, 8)
        self.user_zip_code_embedding = nn.Embedding(user_feature['zip_code'].max() + 1, 18)
        # item embedding
        self.item_id_embedding = nn.Embedding(item_feature['id'].max() + 1, 32)
        self.item_title_embedding = nn.Embedding(item_feature['title'].max() + 1, 8)
        self.item_temperature_embedding = nn.Embedding(item_feature['temperature'].max() + 1, 8)
        self.item_humidity_embedding = nn.Embedding(item_feature['humidity'].max() + 1, 8)
        self.item_windSpeed_embedding = nn.Embedding(item_feature['windSpeed'].max() + 1, 8)
        # 自循环
        self.selfLoop = self.getSelfLoop(self.num_user + self.num_item)
        # 堆叠GCN层
        self.GCN_Layers = torch.nn.ModuleList()
        for _ in range(self.args.gcn_layers):
            self.GCN_Layers.append(GCN_Layer(self.args.embedSize, self.args.embedSize))
        self.graph = self.buildGraph()
        self.transForm = nn.Linear(in_features=self.args.embedSize * (self.args.gcn_layers + 1),
                                   out_features=self.args.embedSize)

    def getSelfLoop(self, num):
        i = torch.LongTensor(
            [[k for k in range(0, num)], [j for j in range(0, num)]])
        val = torch.FloatTensor([1] * num)
        return torch.sparse.FloatTensor(i, val).to(self.device)

    def buildGraph(self):
        rating = self.rating.values
        graph=sp.coo_matrix((rating[:, 2], (rating[:, 0], rating[:, 1])), shape=(self.num_user, self.num_item)).tocsr()
        graph=sp.bmat([[sp.csr_matrix((graph.shape[0], graph.shape[0])), graph],[graph.T, sp.csr_matrix((graph.shape[1], graph.shape[1]))]])
        rom_sum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
        col_sum_sart  = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
        graph = rom_sum_sqrt @ graph @ col_sum_sart
        graph = graph.tocoo()
        values=graph.data
        indices = np.vstack((graph.row, graph.col))
        graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values),torch.Size(graph.shape))
        return graph.to(self.device)
    def getFeature(self):
        # 根据用户特征获取对应的embedding
        user_id = self.user_id_embedding(torch.tensor(self.user_feature['id']).to(self.device))
        age = self.user_age_embedding(torch.tensor(self.user_feature['age']).to(self.device))
        gender = self.user_gender_embedding(torch.tensor(self.user_feature['gender']).to(self.device))
        occupation = self.user_occupation_embedding(torch.tensor(self.user_feature['occupation']).to(self.device))
        location = self.user_zip_code_embedding(torch.tensor(self.user_feature['location']).to(self.device))
        user_emb = torch.cat((user_id, age, gender, occupation, location), dim=1)
        # 根据天气特征获取对应的embedding
        item_id = self.item_id_embedding(torch.tensor(self.item_feature['id']).to(self.device))
        item_type = self.item_title_embedding(torch.tensor(self.item_feature['type']).to(self.device))
        temperature = self.item_temperature_embedding(torch.tensor(self.item_feature['temperature']).to(self.device))
        humidity = self.item_humidity_embedding(torch.tensor(self.item_feature['humidity']).to(self.device))
        windSpeed = self.item_windSpeed_embedding(torch.tensor(self.item_feature['windSpeed']).to(self.device))
        item_emb = torch.cat((item_id, item_type, temperature, humidity, windSpeed), dim=1)
        # 拼接到一起
        concat_emb = torch.cat([user_emb, item_emb], dim=0)
        return concat_emb.to(self.device)

    def forward(self, users, items):
        features=self.getFeature()
        final_emb = features.clone()
        for GCN_Layer in self.GCN_Layers:
            features=GCN_Layer(self.graph,self.selfLoop,features)
            final_emb=torch.cat((final_emb,features.clone()),dim=1)
        user_emb,item_emb=torch.split(final_emb,[self.num_user,self.num_item])
        user_emb=user_emb[users]
        item_emb=item_emb[items]
        user_emb=self.transForm(user_emb)
        item_emb=self.transForm(item_emb)
        prediction=torch.mul(user_emb,item_emb).sum(1)
        return prediction