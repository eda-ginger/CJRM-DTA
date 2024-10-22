import sys
import shlex
import subprocess
from knockknock import discord_sender

# notice discord
webhook_url = "https://discord.com/api/webhooks/1009749385170665533/m4nldXOXR5f9iWaXoCDLNGhNI48XEpy-Y9CcBpdFJW_xUipS54LCzXX9xZaCY6IH0vSl"
@discord_sender(webhook_url=webhook_url)
def finish(message):
    return message  # Optional return value


def split(s):
    params = shlex.split(s)
    print(params)
    return params

## multiple
# # SnS, SnG
cmd_lst = ['--data ./data/preprocessed/KIBA/random --d1_type graph --d2_type seq --data_name KIBA --project_name KIBA_random_GnS_ban --joint bi_att_slience --one_shot --virtual_node --n_workers 4 --batch_size 128 --use_cuda 1',
           '--data ./data/preprocessed/KIBA/random --d1_type seq --d2_type seq --data_name KIBA --project_name KIBA_random_SnS_mgt --joint joint_att_slience --one_shot --n_workers 4 --batch_size 128'
           ]

# '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_GnS_ban --joint bi_att --virtual_node --n_workers 4 --batch_size 256',
# --data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_GnS_ban_server_22 --joint bi_att --virtual_node --n_workers 4 --batch_size 128 --use_cuda 7

# '--data ./data/preprocessed/KIBA/random --d1_type seq --d2_type seq --data_name KIBA --project_name KIBA_random_SnS_ban --joint bi_att --one_shot',


import time
for command in cmd_lst:
    start = time.time()
    finish(f'\nstarting {command}\n - server gpu 0')
    subprocess.run(args=[sys.executable, 'CJRM.py'] + split(command))

    f_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    finish(f'\nFinished: {command}\nLearning time: {f_time} - server gpu 0')
finish(f'\nAll job finished successfully!')


#### done

### DAVIS

## concat
# SnS
# '--data ./data/preprocessed/DAVIS/random --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_random_SnS_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_drug_SnS_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat',
# '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_target_SnS_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_SnS_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat'

# GnS
# '--data ./data/preprocessed/DAVIS/random --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_random_GnS_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_drug_GnS_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat',
# '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_target_GnS_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_GnS_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat'

# SnG
# '--data ./data/preprocessed/DAVIS/random --d1_type seq --d2_type graph --data_name DAVIS --project_name DAVIS_random_SnG_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat',

## multiple
# '--data ./data/preprocessed/DAVIS/random --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_random_SnS_multiple --joint multiple'
# '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_drug_SnS_multiple --joint multiple'
# '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_target_SnS_multiple --joint multiple'
# '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_SnS_multiple --joint multiple'
# '--data ./data/preprocessed/DAVIS/random --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_random_GnS_multiple --joint multiple --use_cuda 1',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_drug_GnS_multiple --joint multiple --use_cuda 1',
# '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_target_GnS_multiple --joint multiple --use_cuda 1',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_GnS_multiple --joint multiple --use_cuda 1',

## Bilinear
# '--data ./data/preprocessed/DAVIS/random --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_random_SnS_bi --joint bi',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_drug_SnS_bi --joint bi',
# '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_target_SnS_bi --joint bi',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_SnS_bi --joint bi',

# '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_GnS_bi --joint bi'
# '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_target_GnS_bi --joint bi --use_cuda 1'

## Addiction
# '--data ./data/preprocessed/DAVIS/random --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_random_SnS_add --joint add',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_drug_SnS_add --joint add',
# '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_target_SnS_add --joint add',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_SnS_add --joint add',
# '--data ./data/preprocessed/DAVIS/random --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_random_GnS_add --joint add',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_drug_GnS_add --joint add',
# '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_target_GnS_add --joint add',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_GnS_add --joint add'

# '--data ./data/preprocessed/DAVIS/random --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_random_SnS_ban --joint bi_att',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_drug_SnS_ban --joint bi_att',
# '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_target_SnS_ban --joint bi_att',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_SnS_ban --joint bi_att',

# '--data ./data/preprocessed/DAVIS/random --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_random_SnS_mgt --joint joint_att',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_drug_SnS_mgt --joint joint_att',
# '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_target_SnS_mgt --joint joint_att',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_SnS_mgt --joint joint_att',

# '--data ./data/preprocessed/DAVIS/random --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_random_SnS_MHA --joint multi_att',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_drug_SnS_MHA --joint multi_att',
# '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_target_SnS_MHA --joint multi_att',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_SnS_MHA --joint multi_att',

# '--data ./data/preprocessed/KIBA/random --d1_type seq --d2_type seq --data_name KIBA --project_name KIBA_random_SnS_MHA --joint multi_att --one_shot',