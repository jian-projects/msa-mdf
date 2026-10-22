import warnings, os, random, torch, sys
import yaml, nni, wandb
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = "false"


from global_var import *
sys.path.append(utils_dir)

from config import config
from writer import JsonFile
from processor import Processor
from utils_processor import set_rng_seed


def run(args, project=None, name=None):    
    ## 0. Initialize
    if args.train['use_wandb']: 
        wandb.init(
            project='mdf' if project is None else project,
            name=f"run_{args.seed}_baseline" if name is None else name,
            entity="msa_ijcai2025"
        )
    set_rng_seed(args.train['seed']) # 

    # import model and dataset
    from Model_MDF_Sent_Old import import_model
    model, dataset = import_model(args)

    processor = Processor(args, model, dataset)
    if args.train['inference']: 
        checkpoint = torch.load(args.model['save_path'])
        processor.model.load_state_dict(checkpoint['net'], strict=False)        
        processor._evaluate(stage='test')
        result = processor.dataset.metrics.results['test']
    else:
        result = processor._train()
        nni.report_final_result({'default': result['test']['mae']})
    
    record = {
        'params': {
            'e':       args.train['epochs'],
            'es':      args.train['early_stop'],
            'lr_pre':  args.train['learning_rate_plm'],
            'bz':      args.train['batch_size'],
            'dr':      args.model['drop_rate'],
            'seed':    args.train['seed'],
        },
        'metric': {
            'val_mae':   result['valid']['mae'],
            'test_mae':  result['test']['mae'],
        },
    }
    print(record['metric'])
    return record


if __name__ == '__main__':
    args = config(task='msa', dataset='mosi', framework=None, model='mdf')

    with open(f"./configs/{args.model['name']}_{args.train['tasks'][-1]}.yaml", 'r') as f:
        run_config = yaml.safe_load(f)
    args.train.update(run_config['train'])
    args.model.update(run_config['model'])
    args.model['w_wei'] = 1.0
    args.model['w_aud'] = 1.0
    args.model['w_ort'] = 1.0

    try:
        tuner_params = nni.get_next_parameter()
        if 'e' in tuner_params:    args.train['epochs'] = tuner_params['e']
        if 'lr' in tuner_params:   args.train['learning_rate_plm'] = tuner_params['lr'] 
        if "bs" in tuner_params:   args.train['batch_size'] = tuner_params['bs']
        if 'seed' in tuner_params: args.train['seed'] = tuner_params['seed']

        if 'w_wei' in tuner_params: args.model['w_wei'] = tuner_params['w_wei']
        if 'w_aud' in tuner_params: args.model['w_aud'] = tuner_params['w_aud']
        if 'w_ort' in tuner_params: args.model['w_ort'] = tuner_params['w_ort']
    except Exception as exception:
        print(exception)
        raise

    seeds = []
    # args.train['inference'] = True  
    recoed_path = f"{args.file['record']}{args.model['name']}_best_4002.jsonl"
    record_show = JsonFile(recoed_path, mode_w='a', delete=False)
    if seeds or args.train['inference']: 
        if not seeds: seeds = [args.train['seed']]
        for seed in seeds:
            args.train['seed'] = seed
            record = run(args)
            record_show.write(record, space=False) 
    else:       
        for c in range(50):
            args.train['seed'] = random.randint(1000,9999)+c
            args.train['seed_change'] = False
            record = run(args)
            record_show.write(record, space=False)