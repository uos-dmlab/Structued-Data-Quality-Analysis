import os
import math
import json
import argparse

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Mapper():
    def __init__(self, X, num_vars, cat_vars):
        self.num_vars = num_vars
        self.cat_vars = cat_vars
        self.get_dict(X, num_vars, cat_vars)
        
    def convert_numeric_feature(self, val):
        if np.isnan(val):
            return 'NULL'
        if val > 2:
            return str(int(math.log(val, 2) ** 2))
        else:
            return str(1)

    def convert_categorical_feature(self, val):
        if np.isnan(val):
            return 'NULL'
        return str(int(val))

    def get_dict(self, X, num_vars, cat_vars):
        f2i = {'OOV': 0}
        for idx in num_vars + cat_vars:
            key = 'COL:{}:NULL'.format(idx)
            f2i[key] = len(f2i)

        for idx in num_vars:
            for v in X[:, idx].astype('float'):
                key = 'COL:{}:{}'.format(idx, self.convert_numeric_feature(v))
                if key not in f2i:
                    f2i[key] = len(f2i)

        for idx in cat_vars:
            for v in X[:, idx].astype('float'):
                key = 'COL:{}:{}'.format(idx, self.convert_categorical_feature(v))
                if key not in f2i:
                    f2i[key] = len(f2i)
        
        self.f2i = f2i
    
    def convert(self, x):
        res = np.zeros(len(self.num_vars)+len(self.cat_vars))
        for idx in self.num_vars:
            key = 'COL:{}:{}'.format(idx, self.convert_numeric_feature(x[idx]))
            res[idx] = self.f2i[key] if key in self.f2i else 0.
        for idx in self.cat_vars:
            key = 'COL:{}:{}'.format(idx, self.convert_categorical_feature(x[idx]))
            res[idx] = self.f2i[key] if key in self.f2i else 0.
        return res.astype('int')
    
    
class MVIDataset(Dataset):
    
    def __init__(self, X, cat_vars, num_vars, scaler, mapper, noise_percent=0):
        self.X = X.astype('float')
        self.n_col = X.shape[1]
        self.cat_vars = cat_vars
        self.num_vars = num_vars
        self.n2i = {n:i for i, n in enumerate(num_vars)}
        self.scaler = scaler
        self.mapper = mapper
        if noise_percent:
            self.n_noise = int(X.shape[1] * noise_percent / 100)
        else:
            self.n_noise = 0
       
    def process_num_vars(self, x):
        x_num = x[self.num_vars]
        x_num = self.scaler.transform(x_num.reshape(1, -1)).squeeze()
        mask_num = np.isnan(x_num)
        x_num[mask_num] = 0. 
        num_tensors = torch.from_numpy(x_num).float()
        mask_num_tensors = torch.tensor(~mask_num, dtype=torch.int)
        return num_tensors, mask_num_tensors

    def process_cat_vars(self, x):
        cat_tensors = [torch.tensor(v, dtype=torch.long) if not np.isnan(v) else torch.tensor(0)
                for v in x[self.cat_vars]]
        mask_cat_tensors = [torch.tensor(~np.isnan(v), dtype=torch.int) for v in x[self.cat_vars]]
        return cat_tensors, mask_cat_tensors

    def get_noise(self, input_num, token_tensors):
        col_idxs = list(range(self.n_col))
        if self.n_noise > 0:
            noise_idxs = sorted(np.random.choice(col_idxs, size=self.n_noise, replace=False))

            for ni in noise_idxs:
                if ni in set(self.num_vars):
                    input_num[self.n2i[ni]] = 0.
                token_tensors[ni] = self.mapper.f2i['COL:{}:NULL'.format(ni)]
        return input_num, token_tensors

    def __getitem__(self, index):
        x = self.X[index]
        tensors = []
        masks = []
        
        num_tensors, mask_num_tensors = self.process_num_vars(x)
        input_num = num_tensors.clone().detach()
        tensors.append(num_tensors)
        masks.append(mask_num_tensors)
        
        cat_tensors, mask_cat_tensors = self.process_cat_vars(x) 
        tensors += cat_tensors
        masks += mask_cat_tensors
        
        token_tensors = self.mapper.convert(x)
        token_tensors = torch.tensor(token_tensors, dtype=torch.long)
        
        input_num, token_tensors = self.get_noise(input_num, token_tensors)
        
        return input_num, token_tensors, tensors, masks
    
    def __len__(self):
        return len(self.X)


class FCUnit(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, bias=True):
        super(FCUnit, self).__init__()
        self.skip_connection = True if input_dim == output_dim else False
        self. fc = nn.Sequential(
                nn.Linear(input_dim, output_dim, bias=bias),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout))
        
        self.apply(self._init_weight)
    
    def _init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        res = self.fc(x)
        if self.skip_connection:
            return res + x
        else:
            return res


class SelfAttentionUnit(nn.Module):
    def __init__(self, embed_dim, num_heads, max_len, dropout=0.8, bias=False, skip_connection=True):
        super(SelfAttentionUnit, self).__init__()
        self.skip_connection = skip_connection
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias)
        self.act = nn.ReLU()
        self.ln = nn.LayerNorm([max_len, embed_dim])

    def forward(self, x):
        x = x.permute(1, 0, 2)
        res, _ = self.attn(key=x, value=x, query=x)
        res = self.act(res)
        if self.skip_connection:
            res = res + x
        res = res.permute(1, 0, 2)
        return self.ln(res)


class Featurize(nn.Module):
    def __init__(self, n_n, n_c, embed_dim, vocab_size, num_heads, dropout, bias=False):
        '''
        input: List of Tensors (num vars and cat vars))
               num_vars: (bsz, len(num_vars))
               token: (bsz, n_col)
        output: featurize (bsz, 1+n_c, rep_dim)
        '''
        super(Featurize, self).__init__()
        n_col = n_n + n_c
        hidden_dim = (1+n_col) * embed_dim
        
        self.numeric_fc = FCUnit(input_dim=n_n,
                output_dim=embed_dim,
                dropout=dropout,
                bias=bias)

        self.embed = nn.Embedding(num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=0)

        self.attn = SelfAttentionUnit(embed_dim=embed_dim,
                num_heads=num_heads,
                max_len=n_col,
                dropout=dropout,
                bias=bias)
        
        self.apply(self._init_weight)
    
    def _init_weight(self, m):
        if isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x_num, tokens):
        f_num = self.numeric_fc(x_num)
        f_emb = self.embed(tokens)
        f_int = self.attn(f_emb).flatten(start_dim=1)
        f_tot = torch.cat([f_num, f_int], dim=1)
        return f_tot


class Model(nn.Module):
    def __init__(self, num_vars, cat_vars, tasks, rep_dim, vocab_size, num_heads, n_hidden, dropout):
        super(Model, self).__init__()

        n_n = len(num_vars)
        n_c = len(cat_vars)
        n_col = n_n + n_c
        hidden_dim = (1+n_col)*rep_dim

        self.featurizer = Featurize(n_n=n_n,
                n_c=n_c,
                embed_dim=rep_dim,
                vocab_size=vocab_size,
                num_heads=num_heads,
                dropout=dropout)
        
        shared_unit = [FCUnit(input_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout)]

        self.shared_unit = nn.Sequential(*shared_unit*n_hidden)

        self.multi_task_mappings = nn.ModuleList()
        self.multi_task_mappings.append(
                nn.Linear(hidden_dim, n_n)
        )

        for t in tasks:
            self.multi_task_mappings.append(
                    nn.Linear(hidden_dim, t if t != 2 else 1)
            )
        
        self.apply(self._init_weight)
    
    def _init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x_num, tokens):
        f_tot = self.featurizer(x_num, tokens)
        z = self.shared_unit(f_tot)
        outputs = []
        for i, task in enumerate(self.multi_task_mappings):
            output = task(z)
            if i > 0 :
                if output.shape[1] == 1:
                    output = torch.sigmoid(output)
                elif output.shape[1] > 1:
                    output = torch.softmax(output, dim=-1)
            outputs.append(output)
        return outputs

                
class Imputer(object):

    def __init__(self, rep_dim=32, num_heads=8, n_hidden=2,
            lr=3e-3, weight_decay=1e-5, batch_size=128,
            epochs=10, noise_percent=30, stopped_epoch=10):
        
        self.rep_dim = rep_dim
        self.num_heads = num_heads
        self.n_hidden = n_hidden
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.stopped_epoch = stopped_epoch
        self.dropout = 0.5
        self.noise_percent = noise_percent

    def cal_loss(self, preds, targets, masks):
        losses = []
        nums = []
        eps = 1e-10
        
        loss_mse = F.mse_loss(preds[0], targets[0], reduction='none')
        loss_mse *= masks[0]
        N = masks[0].sum()
        loss_mse = loss_mse.sum()/(N+eps)
        losses.append(loss_mse)
        nums.append(N)

        for idx, t in enumerate(self.tasks, 1):
            if t == 2: # bce
                loss = F.binary_cross_entropy(input=preds[idx],
                                              target=targets[idx].unsqueeze(-1).float(),
                                              reduction='none').reshape(masks[idx].shape)
            else: # cce
                loss = F.cross_entropy(input=preds[idx],
                                       target=targets[idx],
                                       reduction='none').reshape(masks[idx].shape)
            loss *= masks[idx]
            N = masks[idx].sum()
            loss = loss.sum()/(N+eps)
            losses.append(loss)
            nums.append(N)
        
        losses = torch.stack(losses, 0)
        nums = torch.stack(nums, 0)
        return losses, nums

    def cal_metric_batch(self, preds, targets, masks):
        with torch.no_grad():
            metric = []

            mse = (torch.pow(targets[0] - preds[0], 2) * masks[0]).sum() / masks[0].sum()
            metric.append(mse.item())

            for idx in range(1, len(preds)):
                if preds[idx].shape[1] == 1:
                    res = (preds[idx] > 0.5).to(torch.int).squeeze() == targets[idx]
                else:
                    res = torch.argmax(preds[idx], axis=1) == targets[idx]
                acc = (res * masks[idx]).sum() / masks[idx].sum()
                metric.append(acc.item())
        return metric
    
    def run_batch(self, batch, epoch, train):
        nums, tokens, targets, masks = batch
        tokens = tokens.to(DEVICE)
        nums = nums.to(DEVICE)
        targets = [t.to(DEVICE) for t in targets]
        masks = [m.to(DEVICE) for m in masks]
        batch_size = targets[0].shape[0]

        preds = self.model(nums, tokens)
        losses, nums = self.cal_loss(preds, targets, masks)
        metric = self.cal_metric_batch(preds, targets, masks)
        
        loss_tot = losses.mean()
        
        if train:
            self.optimizer.zero_grad()
            loss_tot.backward()
            self.optimizer.step()

        metric = np.array(metric)
        losses = np.array([l.item() for l in losses])
        nums = np.array([n.item() for n in nums])
        
        return loss_tot.item(), metric, losses, nums, batch_size

    def train(self, epoch=None):
        self.model.train()
        t_loss_tot, cnt = 0., 0.
        n_task = len(self.tasks)+1
        t_metrics = np.zeros(n_task)
        t_losses = np.zeros(n_task)
        mask_cnt = np.zeros(n_task)

        for batch_idx, batch in enumerate(self.train_iter):
            loss_tot, metric, losses, nums, batch_size = self.run_batch(batch, epoch, train=True)
            t_metrics += metric * nums
            t_loss_tot += loss_tot * batch_size
            cnt += batch_size
            t_losses += losses * nums
            mask_cnt += nums
            
        status = {
            'task_metrics' : t_metrics/mask_cnt,
            'task_losses' : t_losses/mask_cnt,
            'total_loss' : t_loss_tot/cnt
        }
        
        return status
    
    def fit(self, X, cat_vars=None):
        
        self.n_col = X.shape[1]
        self.cat_vars = cat_vars if cat_vars else list()
        num_vars = set(range(self.n_col)) - set(self.cat_vars)
        num_vars = list(num_vars)
        self.num_vars = num_vars
        
        self.c2is = None
        scaler = StandardScaler()
        scaler.fit(X[:, self.num_vars])
        mapper = Mapper(X=X, num_vars=num_vars, cat_vars=cat_vars)
        tasks = [int(np.nan_to_num(X[:, idx].astype('float')).max()+1) for idx in cat_vars] 
        
        self.scaler = scaler
        self.mapper = mapper
        self.tasks = tasks

        self.model = Model(num_vars=num_vars,
                cat_vars=cat_vars,
                tasks=tasks,
                rep_dim=self.rep_dim,
                vocab_size=len(mapper.f2i),
                num_heads=self.num_heads,
                n_hidden=self.n_hidden,
                dropout=self.dropout).to(DEVICE)
       
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)

        dataset = MVIDataset(X, cat_vars, num_vars, scaler, mapper, self.noise_percent)
        self.train_iter = DataLoader(dataset=dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=0)
        
        wait = 0
        best_loss = None
        iteration = tqdm(range(1, self.epochs+1))
        
        for epoch in iteration:
            epoch_status = self.train(epoch)
            
            if self.stopped_epoch:
                if best_loss is None or best_loss > epoch_status['total_loss']:
                    best_loss = epoch_status['total_loss']
                    wait = 0
                else:
                    wait += 1
                
                if wait > self.stopped_epoch:
                    print('Terminated Training for Early Stopping at Epoch {}'.format(epoch))
                    break
            
        return self

    def np_inverse_transform(self, array, dic):
        u, inv = np.unique(array, return_inverse=True)
        return np.array([dic[x] for x in u])[inv].reshape(array.shape)
    
    def _transform(self, outputs):
        batch_size = outputs[0].shape[0]
        rec_num_vars = self.scaler.inverse_transform(outputs[0].cpu().detach().numpy())
        rec_cat_vars = []
        for output in outputs[1:]:
            if output.shape[1] == 1:
                output = (output.cpu().detach().numpy() > 0.5).astype('int')
            else:
                output = np.argmax(output.cpu().detach().numpy(), axis=1).reshape(-1, 1)
            rec_cat_vars.append(output.astype('object'))
        rec_cat_vars = np.concatenate(rec_cat_vars, axis=1)

        result = np.zeros((batch_size, self.n_col))
        result[:, self.num_vars] = rec_num_vars
        result = result.astype('object')
        result[:, self.cat_vars] = rec_cat_vars
        return result
    
    def transform(self, X):
        
        dataset = MVIDataset(X, self.cat_vars, self.num_vars, self.scaler, self.mapper)
        impute_iter = DataLoader(dataset=dataset,
                                 batch_size = self.batch_size,
                                 shuffle=False,
                                 num_workers=0)
        
        X_imputed = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(impute_iter):
                nums, tokens, _, _ = batch
                nums = nums.to(DEVICE)
                tokens = tokens.to(DEVICE)
                
                outputs = self.model(nums, tokens)
                result = self._transform(outputs)
                X_imputed.append(result)
        
        return np.vstack(X_imputed)

    def fit_transform(self, X, **fit_params):

        return self.fit(X, **fit_params).transform(X)


class MVImputer(object):

    def __init__(self, input_fname, result_path):
        self.label_encoders = defaultdict(LabelEncoder)
        self.get_data(input_fname)
        self.input_fname = input_fname
        self.result_path = result_path

    def get_data(self, input_fname):
        data = pd.read_csv(input_fname)
        num_idx = data.dtypes[data.dtypes != 'object'].index
        num_vars = [data.columns.get_loc(idx) for idx in num_idx]
        cat_vars = list(set(range(data.shape[1])) - set(num_vars))
        
        self.data = data
        self.num_vars = num_vars
        self.cat_vars = cat_vars
    
    def label_encode(self, data):
        missing = pd.isnull(data.iloc[:, self.cat_vars])
        encode_values = data.iloc[:, self.cat_vars].fillna('NULL')
        encode_values = encode_values.apply(
                lambda x: self.label_encoders[x.name].fit_transform(x)
                )
        encode_values = encode_values.values.astype('float')
        encode_values[missing] = np.nan
        data.iloc[:, self.cat_vars] = encode_values
        return data

    def inverse_transform(self, data):
        inverse_values = data.iloc[:, self.cat_vars].astype('int')
        inverse_values = inverse_values.apply(
                lambda x: self.label_encoders[x.name].inverse_transform(x)
                ) 
        data.iloc[:, self.cat_vars] = inverse_values.values
        return data
    
    def write_json(self, targets):
        obj = {"result": dict()}
        for idx, column in enumerate(self.data.columns):
            obj["result"][column] = int(targets[:, idx].sum())
        obj["input_fname"] = self.input_fname
        obj["result_path"] = self.result_path

        result_json_fname = os.path.join(self.result_path, "result.json")
        with open(result_json_fname, "w") as json_file:
            json.dump(obj, json_file)

    def run(self):

        if not os.path.isdir(self.result_path):
            os.makedirs(self.result_path)

        data = self.label_encode(self.data)
        X_incomplete = data.values
        targets = np.isnan(X_incomplete)
        
        imputer = Imputer()
        X_imputed = imputer.fit_transform(X_incomplete, cat_vars=self.cat_vars)
        imputed_values = X_incomplete.copy()
        imputed_values[targets] = X_imputed[targets]
        result_data = pd.DataFrame(imputed_values, columns=data.columns)
        result_data = self.inverse_transform(result_data)
        
        self.write_json(targets)
        result_csv_fname = os.path.join(self.result_path, "result.csv")
        result_data.to_csv(result_csv_fname, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fname', type=str, default='bank_incomplete.csv')
    parser.add_argument('--result_path', type=str, default='bank_impute')
    args = parser.parse_args()
    
    mvi = MVImputer(input_fname=args.input_fname, result_path=args.result_path)
    mvi.run()
