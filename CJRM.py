########################################################################################################################
########## Sources
########################################################################################################################

# https://github.com/microsoft/Drug-Interaction-Research/tree/DSN-DDI-for-DDI-Prediction
# https://github.com/thinng/GraphDTA

########################################################################################################################
########## Import
########################################################################################################################

import time
import copy
import torch
import logging
import warnings
import argparse
import pandas as pd
from tqdm import tqdm
from torch import optim
from pathlib import Path
from datetime import date
import torch.nn.functional as F
from torch_geometric.data import Data

from models import *
from utils.metric import *
from utils.helper import *
from utils.seq_to_vec import integer_label_string
from utils.seq_to_graph import drug_to_graph, protein_to_graph

########################################################################################################################
########## Pre-settings
########################################################################################################################

tqdm.pandas()
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


########################################################################################################################
########## Functions
########################################################################################################################


# training function at each epoch
def train(model, device, train_loader, loss_fn, optimizer, epoch, scheduler=None):
    start = time.time()
    model.train()

    train_loss = 0
    processed_data = 0
    train_preds = torch.Tensor()
    train_reals = torch.Tensor()
    for batch_idx, data in enumerate(train_loader):

        data = [d.to(device) for d in data]

        processed_data += len(data[2])
        optimizer.zero_grad()

        pred, real = model(data)

        train_preds = torch.cat((train_preds, pred.detach().cpu()), 0)
        train_reals = torch.cat((train_reals, real.detach().cpu()), 0)

        loss = loss_fn(pred, real)
        loss.backward()

        train_loss += loss.item()
        optimizer.step()

        trn_dt_len = len(trn_loader.dataset)
        processed_percent = 100. * processed_data / trn_dt_len
        # if (processed_percent > 50 and log_signal) or (processed_percent == 100):
        if processed_percent == 100:
            runtime = f"{(time.time() - start) / 60:.2f} min"
            logger.info('Train epoch ({}): {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(runtime, epoch,
                                                                                 processed_data,
                                                                                 trn_dt_len,
                                                                                 processed_percent,
                                                                                 loss.item()))
    train_loss = train_loss / len(train_loader)
    if scheduler:
        scheduler.step()
    return train_preds, train_reals, train_loss


def evaluation(model, device, loader):
    model.eval()
    start = time.time()
    total_preds = torch.Tensor()
    total_reals = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = [d.to(device) for d in data]
            pred, real = model(data)

            total_preds = torch.cat((total_preds, pred.detach().cpu()), 0)
            total_reals = torch.cat((total_reals, real.detach().cpu()), 0)

    runtime = f"{(time.time() - start) / 60:.2f} min"
    logger.info(f'eval runtime ({runtime})')
    return total_preds, total_reals


########################################################################################################################
########## Run script
########################################################################################################################

if __name__ == '__main__':

    ####################################################################################################################
    ########## Parameters
    ####################################################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='dataset')
    parser.add_argument('--d1_type', type=str, required=True, help='data1 type')
    parser.add_argument('--d2_type', type=str, required=True, help='data2 type')
    parser.add_argument('--data_name', type=str, required=True, help='dataset name')
    parser.add_argument('--project_name', type=str, required=True, help='project name')
    parser.add_argument('--joint', type=str, required=True, help='choose joint method')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=200, help='num of epochs')
    parser.add_argument('--n_workers', type=int, default=1, help='num of workers for dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--use_cuda', type=int, default=0)
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=0)

    args = parser.parse_args()

    data_folder = Path(args.data)
    d1_type = args.d1_type
    d2_type = args.d2_type
    data_name = args.data_name
    project_name = args.project_name
    joint = args.joint

    lr = args.lr
    n_epochs = args.n_epochs
    n_workers = args.n_workers
    batch_size = args.batch_size
    weight_decay = args.weight_decay

    device = f'cuda:{args.use_cuda}' if torch.cuda.is_available() else 'cpu'
    use_scheduler = args.use_scheduler

    ####################################################################################################################
    ########## Run
    ####################################################################################################################

    # # # tmp
    # data_folder = Path('./data/preprocessed/DAVIS/random')
    # d1_type, d2_type = 'seq', 'seq'
    # d1_type, d2_type = 'seq', 'graph'
    # d1_type, d2_type = 'graph', 'seq'
    # d1_type, d2_type = 'graph', 'graph'
    # data_name = 'davis'
    # project_name = 'Test1'
    # architecture = 'DSN'
    # batch_size = 12
    # lr = 1e-3
    # weight_decay = 5e-4
    # device = 'cpu'
    # n_atom_feats = 55
    # n_atom_hid = 128
    # n_epochs = 2
    # kge_dim = 128
    # n_workers = 0
    # joint = 'concat'

    # output path
    today = str(date.today()).replace('-', '')
    output_folder = Path(f'Results_{project_name}_{today}')
    model_folder = output_folder / 'models'
    model_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"output_folder: {output_folder}")

    # log path
    log_fd = Path(output_folder / 'logs')
    log_fd.mkdir(parents=True, exist_ok=True)
    set_log(log_fd, f'models.log')
    logger.info('Dual-View-Expansion experiments...')
    logger.info(f'data_folder: {data_folder}')
    logger.info(f'd1_type: {d1_type}, d2_type: {d2_type}')
    logger.info(f'data_name: {data_name}')
    logger.info(f'project_name: {project_name}')
    logger.info(f'joint method: {joint}')

    logger.info(f'lr: {lr}')
    logger.info(f'n_workers: {n_workers}')
    logger.info(f'n_epochs: {n_epochs}')
    logger.info(f'batch_size: {batch_size}')
    logger.info(f'weight_decay: {weight_decay}')

    logger.info(f'device: {device}')
    logger.info(f'use_scheduler: {use_scheduler}')

    # load dataset
    df_trn = pd.read_csv(data_folder / 'train.csv')
    df_val = pd.read_csv(data_folder / 'valid.csv')
    df_tst = pd.read_csv(data_folder / 'test.csv')

    df_trn['Set'] = 'trn'
    df_val['Set'] = 'val'
    df_tst['Set'] = 'tst'

    df_total = pd.concat([df_trn, df_val, df_tst]).reset_index(drop=True)

    # preprocess
    if d1_type == 'seq':
        df_total['d1'] = df_total['Drug'].progress_apply(lambda x: integer_label_string(x, 'drug'))
    elif d1_type == 'graph':
        df_total['d1'] = df_total['Drug'].progress_apply(drug_to_graph)
    else:
        raise Exception('d1_type must be either seq or graph')

    if d2_type == 'seq':
        df_total['d2'] = df_total['Target'].progress_apply(lambda x: integer_label_string(x, 'protein'))
    elif d2_type == 'graph':
        p_fd = Path(f'data/preprocessed/{data_name}/protein_graph_pyg')
        p_inform = pd.read_csv(p_fd / f"{data_name}_prot.csv")
        df_total['d2'] = df_total[['Target', 'Target_ID']].progress_apply(
            lambda x: protein_to_graph(x, p_fd, p_inform), axis=1)
    else:
        raise Exception('d2_type must be either seq or graph')

    logger.info(f'None values: {df_total.isnull().sum().sum()}')

    # save
    save_cols = [c for c in df_total.columns if c not in ['d1', 'd2']]
    df_total[save_cols].to_csv(output_folder / f'{project_name}_data.csv', index=False, header=True)
    logger.info(f'Prepare the data: {len(df_total)}')

    df_trn = df_total[df_total['Set'] == 'trn']
    df_val = df_total[df_total['Set'] == 'val']
    df_tst = df_total[df_total['Set'] == 'tst']

    trn_tup = [(h, t, l) for h, t, l in zip(df_trn['d1'], df_trn['d2'], df_trn['Y'])]
    val_tup = [(h, t, l) for h, t, l in zip(df_val['d1'], df_val['d2'], df_val['Y'])]
    tst_tup = [(h, t, l) for h, t, l in zip(df_tst['d1'], df_tst['d2'], df_tst['Y'])]

    # start
    start = time.time()
    total_results = []
    seeds = [5, 42, 76]
    for seed in seeds:
        logger.info(f"#####" * 20)
        train_time = time.time()
        set_random_seeds(seed)

        # Define DataLoader
        trn_dataset = CustomDataset(trn_tup)
        val_dataset = CustomDataset(val_tup)
        tst_dataset = CustomDataset(tst_tup)
        # trn_dataset = CustomDataset(trn_tup[:12])
        # val_dataset = CustomDataset(val_tup[:12])
        # tst_dataset = CustomDataset(tst_tup[:12])
        logger.info(f"TRN: {len(trn_dataset)}, VAL: {len(val_dataset)}, TST: {len(tst_dataset)}")

        trn_loader = CustomDataLoader(trn_dataset, batch_size=batch_size, shuffle=True,
                                      worker_init_fn=seed_worker, num_workers=n_workers)
        val_loader = CustomDataLoader(val_dataset, batch_size=(batch_size * 3),
                                      worker_init_fn=seed_worker, num_workers=n_workers)
        tst_loader = CustomDataLoader(tst_dataset, batch_size=(batch_size * 3),
                                      worker_init_fn=seed_worker, num_workers=n_workers)

        # Define model
        if d1_type == 'seq' and d2_type == 'seq':
            model = SnS()
        elif d1_type == 'seq' and d2_type == 'graph':
            model = SnG()
        elif d1_type == 'graph' and d2_type == 'seq':
            model = GnS()
        elif d1_type == 'graph' and d2_type == 'graph':
            model = GnG()
        else:
            raise Exception('type must be either seq or graph')

        logger.info(f'Model:\n{model}')
        logger.info(f'Model params: {sum(p.numel() for p in model.parameters())}')

        model.to(device)
        loss_fn = nn.MSELoss()
        if use_scheduler:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch)) # on? off?
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)

        # train & evaluation
        best_epoch = -1
        performs = pd.DataFrame()
        model_state = model.state_dict()

        best_loss = np.inf
        model_results = {}
        for epoch in range(n_epochs):
            if use_scheduler:
                trn_preds, trn_reals, trn_loss = train(model, device, trn_loader, loss_fn, optimizer, epoch + 1, scheduler)
            else:
                trn_preds, trn_reals, trn_loss = train(model, device, trn_loader, loss_fn, optimizer, epoch + 1)
            val_preds, val_reals = evaluation(model, device, val_loader)
            val_loss = loss_fn(val_preds, val_reals)

            if val_loss < best_loss:
                # save loss
                best_loss = val_loss
                best_epoch = epoch + 1
                model_state = copy.deepcopy(model.state_dict())

                tst_preds, tst_reals = evaluation(model, device, tst_loader)

                # tst
                model_results['trn'] = (trn_reals, trn_preds)
                model_results['val'] = (val_reals, val_preds)
                model_results['tst'] = (tst_reals, tst_preds)

                logger.info(f"(seed: {seed}) improved at epoch {best_epoch}; best loss: {best_loss}")
            else:
                logger.info(f"(seed: {seed}) No improvement since epoch {best_epoch}; best loss: {best_loss}")

        train_runtime = float(f"{(time.time() - train_time) / 60:.3f}")
        torch.save(model_state, model_folder / f'CJRM_{project_name}_seed{seed}_best.pt')

        trn_perform = cal_perform(model_results['trn'], dt_name='TRN')
        val_perform = cal_perform(model_results['val'], dt_name='VAL')
        tst_perform = cal_perform(model_results['tst'], dt_name='TST')
        performs = pd.DataFrame([trn_perform, val_perform, tst_perform])

        performs['Seed'] = seed
        performs['Project'] = project_name
        performs['Best_epoch'] = best_epoch
        performs['Time (min)'] = train_runtime
        performs.to_csv(model_folder / f'CJRM_{project_name}_seed{seed}_best.csv', header=True, index=False)
        total_results.append(performs)

        logger.info(f'====> (seed: {seed}) best epoch: {best_epoch}; best_loss: {best_loss}')
        logger.info(f"#####" * 20)

    # history
    total_df = pd.concat(total_results).reset_index(drop=True)
    # total_df = pd.read_csv('history.csv')

    # summary - 분산도 필요
    mean_row = []
    for group in total_df.groupby(by=['Set', 'Project']):
        row_dict = {'Set': group[0][0], 'Project': group[0][1]}
        for k, v in group[1].mean(numeric_only=True).to_dict().items():
            if k == 'Seed':
                row_dict['Seeds'] = ', '.join(map(str, group[1]['Seed'].tolist()))
                continue
            elif k == 'Best_epoch':
                row_dict[k] = int(np.ceil(v))
            elif k == 'Time (min)':
                row_dict[k] = f"{v:.2f}"
            else:
                v_std = round(group[1].std(numeric_only=True)[k], 2).item()
                v = f"{v:.4f} ({v_std:.2f})"
                row_dict[k] = v
        mean_row.append(row_dict)

    summary = pd.DataFrame(mean_row)
    total_df.to_csv(output_folder / 'history.csv', index=False, header=True)
    summary.to_csv(output_folder / f'CJRM_{project_name}_summary.csv', index=False, header=True)

    # finish
    runtime = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    logger.info(f"Time : {runtime}")
    logger.info(f'All training jobs done')