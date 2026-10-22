import torch, pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# from moviepy.editor import VideoFileClip


class Metrics(object):
    def __init__(self, base_metric='mae', dataset=None) -> None:
        self.base = base_metric
        self.results = {
            'train': { self.base: 1.0}, 
            'valid': { self.base: 1.0}, 
            'test':  { self.base: 1.0}
            }
        self.dataset = dataset # 可有可无

    def get_class(self, y, eps=1e-8):
        def normalize(arr): return [i/sum(arr) for i in arr]
        classes = {
            'acc_2': normalize([(6 - abs(y+eps - i)) for i in [-3, 3]]), # np.round 直接过拟合
            'acc_7': normalize([(6 - abs(y - i)) for i in range(-3, 4)])
            }

        return classes

    def _score(self, results, eps=1e-8):

        if self.base == 'acc_2': 
            preds = np.concatenate([rec['preds'] for rec in results]).argmax(axis=-1)
            labels = np.concatenate([rec['lab_reg'] for rec in results])
            mark_0 = labels != 0
            return {
                'acc_2': accuracy_score(preds, labels>=0),
                'acc_2_0': accuracy_score(preds[mark_0], (labels>=0)[mark_0]),
                'f1': f1_score(preds, labels>=0, average='weighted'),
                'f1_0': f1_score(preds[mark_0], (labels>=0)[mark_0], average='weighted'),
            }
        
        ## 1. regression
        preds = np.concatenate([rec['preds'] for rec in results])
        labels = np.concatenate([rec['labels'] for rec in results])
        mae, corr = np.mean(np.absolute(preds-labels)), np.corrcoef(preds, labels)[0][1]

        ## 2. classification
        # 2.1 2-class
        def cls_2(y):
            if y < 0-eps: return 0
            else: return 1

        # preds_2, labels_2, mark = (preds+eps) >= 0, (labels+eps) >= 0, labels != 0 # binary classification
        preds_2, labels_2, mark = np.array([cls_2(i) for i in preds]), np.array([cls_2(i) for i in labels]), labels != 0
        acc_2, f1_2 = accuracy_score(preds_2, labels_2), f1_score(preds_2, labels_2, average='weighted')
        acc_2_0, f1_2_0 = accuracy_score(preds_2[mark], labels_2[mark]), f1_score(preds_2[mark], labels_2[mark], average='weighted')

        # 2.2 7-class
        def cls_7(y):
            if y < -2.5+eps: return -3
            elif y < -1.5+eps: return -2
            elif y < -0.5+eps: return -1
            elif y < 0.5+eps: return 0
            elif y < 1.5+eps: return 1
            elif y < 2.5+eps: return 2
            else: return 3

        # preds_7, labels_7 = np.round(np.clip(preds+eps, a_min=-3, a_max=3)), np.round(np.clip(labels+eps, a_min=-3, a_max=3))
        preds_7, labels_7 = np.array([cls_7(i) for i in preds]), np.array([cls_7(i) for i in labels])
        acc_7, f1_7 = accuracy_score(preds_7, labels_7), f1_score(preds_7, labels_7, average='weighted')


        score = {
            'mae': round(mae, 3),
            'corr': round(corr, 3), 
            # 'acc_2': round(acc_2*100, 2),  
            # 'f1_2': round(f1_2*100, 2),
            # 'acc_2_0': round(acc_2_0*100, 2),
            # 'f1_2_0': round(f1_2_0*100, 2),
            # 'acc_7': round(acc_7*100, 2),
        }

        return score


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
        self.datas = {'train': [], 'valid': [], 'test': []}
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
            for i, id in tqdm(enumerate(vanilla['id'])):
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
                sample['classes'] = self.metrics.get_class(sample['label_regress'])
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
            'labels_cls_2': torch.tensor([s['classes']['acc_2'] for s in samples], dtype=torch.float32),
            'labels_cls_7': torch.tensor([s['classes']['acc_7'] for s in samples], dtype=torch.float32),
            'labels_regress': torch.tensor([s['label_regress'] for s in samples], dtype=torch.float32),
        }

        return batch


if __name__ == "__main__":
    # 0. 获取文本数据
    path = f'./datasets/msa/mosi/'
    dataset = MSADataModule(path)

    # 1. 获取向量数据
    model_name = 'Y:/Pretrained_Models//bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset.get_vector(tokenizer, truncate='tail') # 裁剪尾部