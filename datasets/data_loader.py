import torch, pickle
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score


class Metrics(object):
    def __init__(self, base_metric='mae', dataset=None) -> None:
        self.base = base_metric
        self.results = {
            'train': { self.base: 1.0}, 
            'valid': { self.base: 1.0}, 
            'test':  { self.base: 1.0}
            }
        self.dataset = dataset # 可有可无

    def _score(self, results, stage='train'):

        preds = np.concatenate([rec['regress_p'] for rec in results])
        labels = np.concatenate([rec['regress_l'] for rec in results])
        mae, corr = np.mean(np.absolute(preds-labels)), np.corrcoef(preds, labels)[0][1]

        # metric after value round 
        acc_i, f1_i = {3: 0, 2: 0, 1: 0}, {3: 0, 2: 0, 1: 0}
        for clip in [3, 2, 1]: 
            pre_tmp, lab_tmp = np.round(np.clip(preds, a_min=-clip, a_max=clip)), np.round(np.clip(labels, a_min=-clip, a_max=clip))
            acc_i[clip], f1_i[clip] = round(accuracy_score(pre_tmp, lab_tmp), 2), round(f1_score(pre_tmp, lab_tmp, average='weighted'), 2)

        # metric of binary
        acc_b, f1_b = {'w0': 0, 'w_0': 0}, {'w0': 0, 'w_0': 0} 
        for key in ['w0', 'w_0']: # w0: negative/non-negative; w_0: negative/positive
            if key == 'w0':
                zero_mark = np.arange(len(labels))
                pre_tmp, lab_tmp = preds[zero_mark] >= 0, labels[zero_mark] >= 0
            else:
                zero_mark = np.array([i for i,v in enumerate(labels) if v])
                pre_tmp, lab_tmp = preds[zero_mark] > 0, labels[zero_mark] > 0
            acc_b[key], f1_b[key] = round(accuracy_score(pre_tmp, lab_tmp), 2), round(f1_score(pre_tmp, lab_tmp, average='weighted'), 2)
            
        score = {
            'mae': round(mae, 4),
            # 'corr': round(corr, 4), 
            'acc': round(acc_b['w0']*100, 2),
            # 'f1_2': f1_i[1],
            
            # 'accuracy': {
            #     'integer': acc_i,
            #     'binary': acc_b,
            # },
            # 'f1_score': {
            #     'integer': f1_i,
            #     'binary': f1_b,
            # }
            # 'accuracy': acc_b,
            # 'f1_score': f1_b,
        }

        return score



        preds = np.concatenate([rec['preds'].cpu().numpy() for rec in results])
        truthes = np.concatenate([rec['labels'].cpu().numpy() for rec in results])
        losses = [rec['loss'].item() for rec in results]

        score_f1 = round(f1_score(truthes, preds, average='macro'), 4)
        score_acc = round(accuracy_score(truthes, preds), 4)
        score_loss = round(sum(losses)/len(losses), 4)

        return {
            'mae'  : score_f1,
            'corr' : score_acc,

            'loss': score_loss
        }



class MSADataModule(object):
    def __init__(self, data_dir, batch_size=1, data_type='aligned', num_workers=8):
        super().__init__()
        self.name = ['msa', data_dir.split('/')[-2]]
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = 0 # num_workers
        self.data_type = data_type

        self.dataset_init() # dataset initialize
        self.prepare_data(stages=['train', 'valid', 'test']) # 
        self.metrics = Metrics(base_metric='mae')

    def dataset_init(self):
        self.datas = {'train': []}
        self.loader = {'train': None, 'valid': None, 'test': None}


    def info_load(self):
        with open(self.data_dir+f'{self.data_type}_50.pkl', 'rb') as fr:
            raw_process_data_all = pickle.load(fr)  
        return raw_process_data_all
     
    def prepare_data(self, stages=['train', 'valid', 'test']):
        with open(self.data_dir+f'{self.data_type}_50.pkl', 'rb') as fr:
            vanilla_data = pickle.load(fr)  

        for stage in stages:
            vanilla, samples = vanilla_data[stage], []
            for i, id in enumerate(vanilla['id']):
                length = np.sum(vanilla['text_bert'][i][1])
                sample = {
                    'idx': len(samples),
                    'id_name': id,
                    'text': vanilla['raw_text'][i].lower(),
                    # 'text_len': text_len,
                    'bert_ids': vanilla['text_bert'][i],
                    'bert_vec': vanilla['text'][i],
                    'bert_len': length,
                    
                    'audio_vec': vanilla['audio'][i], # [50, 5]: 50 is padding, 5 is dimension
                    'audio_len': vanilla['audio_lengths'][i] if self.data_type=='unaligned' else 50,

                    'vision_vec': vanilla['vision'][i], # [50, 20]: 50 is padding, 20 is dimension
                    'vision_len': vanilla['vision_lengths'][i] if self.data_type=='unaligned' else 50,

                    'polarity': vanilla['annotations'][i].lower(),
                    'label_class': vanilla['classification_labels'][i],
                    'label_regress': vanilla['regression_labels'][i],
                }   
                samples.append(sample)

            self.datas[stage] = samples

    def setup(self, stage=None):
        for stage, samples in self.datas.items():
            for sample in samples:
                embedding = self.tokenizer(sample['text'], return_tensors='pt', padding=True)
                sample['input_ids'] = embedding['input_ids'].squeeze(0)
                sample['attention_mask'] = embedding['attention_mask'].squeeze(0)

        self.loader['train'] = self.train_dataloader()
        self.loader['valid'] = self.val_dataloader()
        self.loader['test'] = self.test_dataloader()
        
    def train_dataloader(self):
        return DataLoader(
            self.datas['train'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.datas['valid'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datas['test'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, samples):
        batch = {
            'idx': torch.tensor([s['idx'] for s in samples]),
            'input_ids': pad_sequence([s['input_ids'] for s in samples], batch_first=True, padding_value=self.tokenizer.pad_token_id),
            'attention_mask': pad_sequence([s['attention_mask'] for s in samples], batch_first=True, padding_value=0),
            'text_vec': torch.tensor([s['bert_vec'] for s in samples], dtype=torch.float32),
            'text_len': torch.tensor([s['bert_len'] for s in samples]),
            'audio_vec': torch.tensor([s['audio_vec'] for s in samples], dtype=torch.float32),
            'audio_len': torch.tensor([s['audio_len'] for s in samples]),
            'vision_vec': torch.tensor([s['vision_vec'] for s in samples], dtype=torch.float32),
            'vision_len': torch.tensor([s['vision_len'] for s in samples]),
            'labels_class': torch.tensor([s['label_class'] for s in samples], dtype=torch.int64),
            'labels_regress': torch.tensor([s['label_regress'] for s in samples], dtype=torch.float32),
        }

        return batch


if __name__ == "__main__":
    # 0. 获取文本数据
    path = f'./mosi/'
    dataset = MSADataModule(path)

    # 1. 获取向量数据
    model_name = 'Y:/Pretrained_Models//bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset.get_vector(tokenizer, truncate='tail') # 裁剪尾部