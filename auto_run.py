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


cmd_lst = ['--data ./data/preprocesseed/DAVIS/random --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_random_ss --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024',

           '--data ./data/preprocesseed/DAVIS/cold_split/Drug --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name DAVIS --task_name DTA --project_name DAVIS_cold_drug_DSN --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --architecture DSN',
           '--data ./data/preprocesseed/DAVIS/cold_split/Target --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name DAVIS --task_name DTA --project_name DAVIS_cold_target_DSN --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --architecture DSN',
           '--data ./data/preprocesseed/DAVIS/cold_split/Drug_and_Target --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name DAVIS --task_name DTA --project_name DAVIS_cold_target_DSN --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --architecture DSN',
           '--data ./data/preprocesseed/KIBA/random --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name KIBA --task_name DTA --project_name KIBA_random_DSN --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --architecture DSN',
           '--data ./data/preprocesseed/KIBA/cold_split/Drug --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name KIBA --task_name DTA --project_name KIBA_cold_drug_DSN --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --architecture DSN',
           '--data ./data/preprocesseed/KIBA/cold_split/Target --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name KIBA --task_name DTA --project_name KIBA_cold_target_DSN --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --architecture DSN'
           ]

import time
for command in cmd_lst:
    start = time.time()
    finish(f'\nstarting {command}\n')
    subprocess.run(args=[sys.executable, 'CJRM.py'] + split(command))

    f_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    finish(f'\nFinished: {command}\nLearning time: {f_time}')
finish(f'\nAll job finished successfully!')
