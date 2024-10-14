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
import utils
import torch
import models
import logging
import warnings
import argparse
import pandas as pd
from tqdm import tqdm
from torch import optim
from pathlib import Path
from datetime import date
import torch.nn.functional as F

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

        processed_data += len(data[0])
        optimizer.zero_grad()

        pred, real = model(data)

        train_preds = torch.cat((train_preds, pred.cpu()), 0)
        train_reals = torch.cat((train_reals, real.cpu()), 0)

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


def evaluation(model, device, loader, set_name):
    model.eval()
    start = time.time()
    total_preds = torch.Tensor()
    total_reals = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = [d.to(device) for d in data]
            pred, real = model(data)

            total_preds = torch.cat((total_preds, pred.cpu()), 0)
            total_reals = torch.cat((total_reals, real.cpu()), 0)

    perform = cal_perform(total_reals, total_preds, set_name)
    runtime = f"{(time.time() - start) / 60:.2f} min"
    logger.info(f'eval runtime ({runtime})')
    return perform


########################################################################################################################
########## Run script
########################################################################################################################

if __name__ == '__main__':

    ####################################################################################################################
    ########## Parameters
    ####################################################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='dataset')
    parser.add_argument('--d1_col', type=str, required=True, help='data1 column name')
    parser.add_argument('--d2_col', type=str, required=True, help='data2 column name')
    parser.add_argument('--d1_type', type=str, required=True, help='data1 type')
    parser.add_argument('--d2_type', type=str, required=True, help='data2 type')
    parser.add_argument('--data_name', type=str, required=True, help='dataset name')
    parser.add_argument('--task_name', type=str, required=True, help='task (DTA or PPI)')
    parser.add_argument('--project_name', type=str, required=True, help='project name')
    parser.add_argument('--architecture', type=str, required=True, help='choose architecture (DSN, DSN-joint)')

    parser.add_argument('--n_atom_feats', type=int, default=55, help='num of input features')
    parser.add_argument('--n_atom_hid', type=int, default=128, help='num of hidden features')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=200, help='num of epochs')
    parser.add_argument('--n_workers', type=int, default=1, help='num of workers for dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--use_cuda', type=int, default=0)

    args = parser.parse_args()

    data_folder = Path(args.data)
    d1_col = args.d1_col
    d2_col = args.d2_col
    d1_type = args.d1_type
    d2_type = args.d2_type
    data_name = args.data_name
    task_name = args.task_name
    project_name = args.project_name
    architecture = args.architecture

    n_atom_feats = args.n_atom_feats
    n_atom_hid = args.n_atom_hid
    lr = args.lr
    n_epochs = args.n_epochs
    n_workers = args.n_workers
    batch_size = args.batch_size

    weight_decay = args.weight_decay
    device = f'cuda:{args.use_cuda}' if torch.cuda.is_available() else 'cpu'

    ####################################################################################################################
    ########## Run
    ####################################################################################################################

    # # # tmp
    # data_folder = Path('./TDC/DTA/DAVIS/random')
    # d1_type, d2_type = 'seq', 'seq'
    # data_name = 'davis'
    # task_name = 'DTA'
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
    # n_workers = 1

    # output path
    today = str(date.today()).replace('-', '')
    output_folder = Path(f'Results_{project_name}_{today}')
    model_folder = output_folder / 'models'
    model_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"output_folder: {output_folder}")

    # log path
    log_fd = Path(output_folder / 'logs')
    log_fd.mkdir(parents=True, exist_ok=True)
    utils.set_log(log_fd, f'models.log')
    logger.info('Dual-View-Expansion experiments...')
    logger.info(f'data_folder: {data_folder}')
    logger.info(f'd1_type: {d1_type}, d2_type: {d2_type}')
    logger.info(f'data_name: {data_name}')
    logger.info(f'task_name: {task_name}')
    logger.info(f'project_name: {project_name}')
    logger.info(f'architecture: {architecture}')
    logger.info(f'batch_size: {batch_size}')
    logger.info(f'lr: {lr}')
    logger.info(f'device: {device}')
    logger.info(f'weight_decay: {weight_decay}')
    logger.info(f'n_atom_feats: {n_atom_feats}')
    logger.info(f'n_atom_hid: {n_atom_hid}')
    logger.info(f'n_workers: {n_workers}')
    logger.info(f'n_epochs: {n_epochs}')

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
        df_total['d1'] = df_total['Drug'].progress_apply(integer_label_string)
    elif d1_type == 'graph':
        df_total['d1'] = df_total['Drug'].progress_apply(drug_to_graph)
    else:
        raise Exception('d1_type must be either seq or graph')

    if d2_type == 'seq':
        df_total['d2'] = df_total['Target'].progress_apply(integer_label_string)
    elif d2_type == 'graph':
        p_fd = Path(f'TDC/DTA/{data_name}/protein_graph_pyg')
        p_inform = pd.read_csv(p_fd / f"{data_name}_prot.csv")
        df_total['d2'] = df_total[['Target', 'Target_ID']].progress_apply(
            lambda x: protein_to_graph(x, p_fd, p_inform), axis=1)
    else:
        raise Exception('d2_type must be either seq or graph')

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
        set_random_seeds(seed)

        # Define DataLoader
        trn_dataset = CustomDataset(trn_tup, shuffle=True)
        val_dataset = CustomDataset(val_tup)
        tst_dataset = CustomDataset(tst_tup)
        # trn_dataset = CustomDataset(trn_tup[:12], shuffle=True)
        # val_dataset = CustomDataset(val_tup[:12])
        # tst_dataset = CustomDataset(tst_tup[:12])
        logger.info(f"TRN: {len(trn_dataset)}, VAL: {len(val_dataset)}, TST: {len(tst_dataset)}")

        trn_loader = CustomDataLoader(trn_dataset, batch_size=batch_size, shuffle=True,
                                      worker_init_fn=utils.seed_worker, num_workers=n_workers)
        val_loader = CustomDataLoader(val_dataset, batch_size=(batch_size * 3),
                                      worker_init_fn=utils.seed_worker, num_workers=n_workers)
        tst_loader = CustomDataLoader(tst_dataset, batch_size=(batch_size * 3),
                                      worker_init_fn=utils.seed_worker, num_workers=n_workers)

        # Define model
        model = models.MVN_DDI(n_atom_feats, n_atom_hid, kge_dim, heads_out_feat_params=[64, 64, 64, 64],
                               blocks_params=[2, 2, 2, 2], arch=architecture)

        model.to(device)

        logger.info(f'Model:\n{model}')
        logger.info(f'Model params: {sum(p.numel() for p in model.parameters())}')
        loss_fn = F.mse_loss
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch)) # on? off?

        # train & evaluation
        best_epoch = -1
        performs = pd.DataFrame()
        model_state = model.state_dict()

        best_loss = np.inf
        model_results = {}
        for epoch in range(n_epochs):
            trn_preds, trn_reals, trn_loss = train(model, device, trn_loader, loss_fn, optimizer, epoch + 1, scheduler)
            check_perform = evaluation(model, device, val_loader, set_name='VAL')
            val_loss = check_perform['MSE']

            if val_loss < best_loss:
                # save loss
                best_loss = val_loss
                best_epoch = epoch + 1
                model_state = copy.deepcopy(model.state_dict())

                # tst
                model_results['trn'] = cal_perform(trn_reals, trn_preds, dt_name='TRN')
                model_results['val'] = check_perform
                model_results['tst'] = evaluation(model, device, tst_loader, set_name='VAL')

                logger.info(f"(seed: {seed}) improved at epoch {best_epoch}; best loss: {best_loss}")
            else:
                logger.info(f"(seed: {seed}) No improvement since epoch {best_epoch}; best loss: {best_loss}")

        torch.save(model_state, model_folder / f'CJRM_{project_name}_seed{seed}_best.pt')

        trn_perform = model_results['trn']
        val_perform = model_results['val']
        tst_perform = model_results['tst']
        performs = pd.DataFrame([trn_perform, val_perform, tst_perform])

        performs['Seed'] = seed
        performs['Task'] = task_name
        performs['Project'] = project_name
        performs['Best_epoch'] = best_epoch
        performs.to_csv(model_folder / f'CJRM_{project_name}_seed{seed}_best.csv', header=True, index=False)
        total_results.append(performs)

        logger.info(f'====> (seed: {seed}) best epoch: {best_epoch}; best_loss: {best_loss}')
        logger.info(f"#####" * 20)

    # history
    total_df = pd.concat(total_results).reset_index(drop=True)
    total_df.to_csv(output_folder / 'history.csv', index=False, header=True)

    # summary - 분산도 필요
    mean_row = []
    for group in total_df.groupby(by=['Set', 'Task', 'Project']):
        row_dict = {'Set': group[0][0], 'Task': group[0][1], 'Project': group[0][2]}
        for k, v in group[1].mean(numeric_only=True).to_dict().items():
            if k == 'Seed':
                row_dict['Seeds'] = len(group[1])
                continue
            elif k == 'Best_epoch':
                v = int(np.ceil(v))
            row_dict[k] = v
        mean_row.append(row_dict)

    summary = pd.DataFrame(mean_row)
    summary.to_csv(output_folder / f'{task_name}_{project_name}_summary.csv', index=False, header=True)

    # finish
    runtime = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    logger.info(f"Time : {runtime}")
    logger.info(f'All training jobs done')