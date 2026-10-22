import torch, os, json, nni
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer    

from utils_rnn import *
from data_loader_old import *
from utils_model import *


def config_for_model(args, scale='base'):
    scale = args.model['scale']
    if scale == 'base': args.model['plm'] = args.file['plm_dir'] + f'bert-{scale}-uncased'
    if scale == 'large': args.model['plm'] = args.file['plm_dir'] + f'bert-{scale}'

    args.model['data_type'] = 'unaligned'
    args.model['data'] = args.file['cache_dir']+f"{args.model['name']}_dataset_{args.model['data_type']}.pt"
    args.model['save_path'] = args.file['cache_dir']+f"{args.model['name']}_{args.train['tasks'][-1]}_checkpoint.pt"
    # args.model['optim_sched'] = ['Adam_', None]

    args.model['video_plm'] = args.file['plm_dir'] + f'Video/videomae-{scale}'
    args.model['audio_plm'] = args.file['plm_dir'] + f'Audio/wav2vec2-{scale}'

    return args

def import_model(args):
    args = config_for_model(args) 
    
    data_path = args.model['data']
    if os.path.exists(data_path):
        dataset = torch.load(data_path)
    else:
        dataset = MSADataModule(
            args.file['data_dir']+f"{args.train['tasks'][-1]}/", 
            data_type=args.model['data_type']
            )
        # torch.save(dataset, data_path)

    model = MDF(
        args=args,
        dataset=dataset,
    )
    return model, dataset


class MDF(ModelForClassification):
    def __init__(self, args, dataset, plm=None):
        super(MDF, self).__init__()
        self.args = args
        self.dataset = dataset

        self.plm_model = AutoModel.from_pretrained(plm if plm is not None else args.model['plm'])
        self.plm_pooler = PoolerAll(self.plm_model.config) 
        self.hidden_dim = self.plm_model.config.hidden_size

        aud_dim = 5 if dataset.name[-1] == 'mosi' else 74
        vis_dim = 20 if dataset.name[-1] == 'mosi' else 35

        self.lstm_aud = DynamicLSTM(aud_dim, aud_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.align_audio = nn.Linear(aud_dim, self.hidden_dim)
        # self.align_audio = nn.Sequential(
        #     nn.Linear(5, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.hidden_dim),
        # )

        self.lstm_vis = DynamicLSTM(vis_dim, vis_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.align_vision = nn.Linear(vis_dim, self.hidden_dim)
        # self.align_vision = nn.Sequential(
        #     nn.Linear(20, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.hidden_dim),
        # )

        self.distangle_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.distangle_v = nn.Linear(self.hidden_dim, self.hidden_dim)
            
        self.weight = nn.Linear(self.hidden_dim*3, 3)
        # self.weight = nn.Sequential(
        #     nn.Linear(self.hidden_dim*3, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 3)
        # )
        self.regress = nn.Linear(self.hidden_dim, 1)
        # self.regress = nn.Sequential(
        #     nn.Linear(self.hidden_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1)
        # )
        
        # self.class_2 = nn.Linear(self.hidden_dim, 2)
        # self.class_7 = nn.Linear(self.hidden_dim, 7)

        self.dropout = nn.Dropout(args.model['drop_rate'])
        self.loss_mse = torch.nn.MSELoss(reduction='mean').cuda()
        self.loss_mae = torch.nn.L1Loss(reduction='mean').cuda()
        
    def epoch_deal(self, epoch=0):
        pass

    def forward(self, inputs, stage='train'):        
        ## 1. text modality
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        txt_out = self.plm_model(
            input_ids, 
            attention_mask=attention_mask
            )#.pooler_output
        txt = self.plm_pooler(txt_out.last_hidden_state)[:,0]
        txt = self.dropout(txt)
        # txt = self.dropout(txt_out.pooler_output)

        ## 2. audio modality
        _, (aud_vec, _) = self.lstm_aud(inputs['audio_vec'], inputs['audio_len']) # .reshape(len(txt), -1)
        aud_ = self.align_audio(aud_vec.mean(dim=0))
        aud_ = self.dropout(aud_)
        # 2.1 decouple
        aud_txt = self.dropout(self.distangle_a(aud_))
        aud = aud_ - aud_txt
        # 2.2 align loss
        loss_aud = self.loss_mse(self.regress(aud_txt).squeeze(), self.regress(txt).squeeze())

        ## 3. vision modality
        # vis_vec = inputs['vision_vec'].reshape(len(txt), -1)
        _, (vis_vec, _) = self.lstm_vis(inputs['vision_vec'], inputs['vision_len']) 
        vis = self.align_vision(vis_vec.mean(dim=0))
        vis = self.dropout(vis)
        
        # # 3.1 decouple
        # vis_txt = self.distangle_v(vis)
        # vis = vis - vis_txt
        # # 3.2 align loss
        # loss_vis = self.loss_mse(self.regress(vis_txt).squeeze(), self.regress(txt).squeeze())
        # loss_aud = loss_aud + loss_vis

        ## 4. Fusion & Prediction
        loss_orth = self.orthogonal_loss(txt, aud, vis)      

        fusion = torch.stack([txt, aud, vis])
        feat_con = fusion.transpose(1, 0).reshape(len(txt), -1)
        weights = self.weight(feat_con)
        weights[weights<=0] = 0
        weights = F.softmax(weights, dim=1)
        # weights = torch.tensor([[1,0,0]]*len(weights)).type_as(weights)
        feat_fuse = torch.sum(fusion*weights.transpose(1,0).unsqueeze(dim=-1), dim=0)
        feat_fuse = self.dropout(feat_fuse)

        y = inputs['labels_regress']
        y_t, y_a, y_v = self.regress(txt).squeeze(), self.regress(aud).squeeze(), self.regress(vis).squeeze()
        w_t, w_a, w_v = 1/(torch.abs(y-y_t)+1e-8), 1/(torch.abs(y-y_a)+1e-8), 1/(torch.abs(y-y_v)+1e-8)
        wei_star = F.softmax(torch.stack([w_t, w_a, w_v], dim=1), dim=1)
        loss_wei = self.loss_mse(weights, wei_star)

        if self.dataset.metrics.base == 'mae':
            preds = self.regress(feat_fuse).squeeze()
            labels = inputs['labels_regress']
            loss = self.loss_mse(preds, labels)
        elif self.dataset.metrics.base == 'acc_2':
            logits = self.class_2(feat_fuse)
            preds, labels = F.softmax(logits, dim=1), inputs['labels_cls_2']
            loss = -torch.mean(torch.einsum('ij,ij->i', torch.log(preds), labels))
        elif self.dataset.metrics.base == 'acc_7':
            logits = self.class_7(feat_fuse)
            preds, labels = F.softmax(logits, dim=1), inputs['labels_cls_7']
            loss = -torch.mean(torch.einsum('ij,ij->i', torch.log(preds), labels))

        loss_total = loss + loss_aud*self.args.model['w_aud'] + loss_orth*self.args.model['w_ort'] + loss_wei*self.args.model['w_wei']

        if self.args.train['inference']:
            return {
                'loss': loss_total,
                'preds': preds.detach().cpu().numpy(),
                'labels': labels.detach().cpu().numpy(),
                
                'index': inputs['idx'].detach().cpu(),

                'txt_ori': txt.detach().cpu(),
                'aud_ori': aud_.detach().cpu(),
                'vis_ori': vis.detach().cpu(),

                'aud_txt': aud_txt.detach().cpu(),
                'aud_sou': aud.detach().cpu(),

                'wei_star': wei_star.detach().cpu(),
                'weights': weights.detach().cpu(),
                'fusion': feat_fuse.detach().cpu(),
            }

        return {
            'loss': loss_total,
            'preds': preds.detach().cpu().numpy(),
            'labels': labels.detach().cpu().numpy(),
            'lab_reg': inputs['labels_regress'].detach().cpu().numpy(),
        }

    def orthogonal_loss(self, F1_norm, F2_norm, F3_norm):
        F1_norm = F.normalize(F1_norm, dim=-1)
        F2_norm = F.normalize(F2_norm, dim=-1)
        F3_norm = F.normalize(F3_norm, dim=-1)

        # loss_11 = self.loss_mse(F1, F1_norm)
        # loss_22 = self.loss_mse(F2, F2_norm)
        # loss_33 = self.loss_mse(F3, F3_norm)

        loss_12 = torch.sum((F1_norm @ F2_norm.T) ** 2) / F1_norm.size(0)
        loss_13 = torch.sum((F1_norm @ F3_norm.T) ** 2) / F1_norm.size(0)
        loss_23 = torch.sum((F2_norm @ F3_norm.T) ** 2) / F1_norm.size(0)

        loss = loss_12 + loss_13 + loss_23 # + loss_11 + loss_22 + loss_33
        return loss

    
    def on_train_epoch_end(self):
        outputs, metrics_tr = self.training_step_outputs, self.dataset.metrics.results['train']
        metrics_tr.update(self.dataset.metrics._score(outputs, eps=self.args.model['eps']))
        metrics_tr['loss'] = round(torch.stack([output['loss'] for output in outputs]).mean().item(),3)
        metrics_tr['epoch'] = self.cur_epoch
        
        self.training_step_outputs = [] # init record
        describe = json.dumps({k: round(float(v),4) for k,v in metrics_tr.items()})
        self.args.logger['process'].info(f"train_eval: {describe}")

    def on_validation_end(self):
        outputs, metrics_vl = self.validation_step_outputs, self.dataset.metrics.results['valid']
        metrics = self.dataset.metrics._score(outputs, eps=self.args.model['eps'])

        mark, self.valid_update = self.dataset.metrics.base, False
        update = metrics[mark]>metrics_vl[mark] if mark!='mae' else metrics[mark]<metrics_vl[mark]
        if update: # Performance improved
            metrics_vl.update(metrics)
            metrics_vl['epoch'] = self.cur_epoch
            describe = json.dumps({k: round(float(v),4) for k,v in metrics_vl.items()})
            self.args.logger['process'].info(f"valid: {describe}")
            self.valid_update = True # execute test

            if self.args.train['save_model']: 
                if self.args.model['use_adapter']: 
                    self.save_checkpoint_peft()
                else: self.save_checkpoint() #

        self.validation_step_outputs = [] # init record
        if self.args.train['use_nni']: 
            met = self.dataset.metrics.base
            nni.report_intermediate_result({
                'default':      self.dataset.metrics.results['test'][met],
                f'train_{met}': self.dataset.metrics.results['train'][met],
                f'valid_{met}': self.dataset.metrics.results['valid'][met],
                f'test_{met}':  self.dataset.metrics.results['test'][met]
                })
            
        if self.args.train['tasks'][-1] == 'mosi':
            if self.cur_epoch > 5 and metrics_vl['mae'] >=0.99:
                metrics_vl['epoch'] = -100
            if self.cur_epoch > 15 and metrics_vl['mae'] > 0.8:
                metrics_vl['epoch'] = -100
        else:
            if self.cur_epoch > 3 and metrics_vl['mae'] >=0.6:
                metrics_vl['epoch'] = -100

    def on_test_end(self):
        outputs, metrics_te = self.test_step_outputs, self.dataset.metrics.results['test']
        metrics_te.update(self.dataset.metrics._score(outputs, eps=self.args.model['eps']))
        metrics_te['epoch'] = self.cur_epoch
        
        self.test_step_outputs = []
        describe = json.dumps({k: round(float(v),4) for k,v in metrics_te.items()})
        self.args.logger['process'].info(f"test: {describe}")

        if self.args.train['inference']:
            torch.save(outputs, './checkpoints/embeddings.pt')

    def save_checkpoint(self, save_path=None, mode='save'):
        if save_path is None: save_path = self.args.model['save_path']
        if mode == 'save':
            state = {
                'net': self.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(state, save_path)

        if mode == 'load':
            state = torch.load(save_path)
            self.load_state_dict(state['net'])
            self.optimizer.load_state_dict(state['optimizer'])

    def save_checkpoint_peft(self, save_path=None, mode='save'):
        if save_path is None: save_path = self.args.model['save_path']
        if mode == 'save':
            training_params_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
            state = {
                'net': training_params_dict,
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(state, save_path)

        if mode == 'load':
            state = torch.load(save_path)
            self.load_state_dict(state['net'], strict=False)
            self.optimizer.load_state_dict(state['optimizer'])


