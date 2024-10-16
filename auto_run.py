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


# GnS
cmd_lst = ['--data ./data/preprocessed/DAVIS/random --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_random_GnS_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat',
           '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_drug_GnS_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat',
           '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_target_GnS_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat',
           '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_GnS_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat'
           ]

# # SnS, SnG
cmd_lst = ['--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_GnS_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat,'
           
           '--data ./data/preprocessed/DAVIS/random --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_random_SnS_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat',
           '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_drug_SnS_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat',
           '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_target_SnS_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat',
           '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_SnS_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat'

           '--data ./data/preprocessed/DAVIS/random --d1_type seq --d2_type graph --data_name DAVIS --project_name DAVIS_random_SnG_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat',
           '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type seq --d2_type graph --data_name DAVIS --project_name DAVIS_cold_drug_SnG_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat',
           '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type seq --d2_type graph --data_name DAVIS --project_name DAVIS_cold_target_SnG_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat',
           '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type seq --d2_type graph --data_name DAVIS --project_name DAVIS_cold_DT_SnG_concat --n_workers 10 --n_epochs 1000 --lr 1e-4 --batch_size 1024 --joint concat'
           ]

# cmd_lst = ['--data ./data/preprocessed/DAVIS/random --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_random_SnS_concat --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --joint concat',
#            '--data ./data/preprocessed/DAVIS/random --d1_type seq --d2_type graph --data_name DAVIS --project_name DAVIS_random_SnG_concat --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --joint concat',
#            '--data ./data/preprocessed/DAVIS/random --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_random_GnS_concat --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --joint concat',
#            '--data ./data/preprocessed/DAVIS/random --d1_type graph --d2_type graph --data_name DAVIS --project_name DAVIS_random_GnG_concat --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --joint concat',
#
#            '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_drug_SnS_concat --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --joint concat',
#            '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type seq --d2_type graph --data_name DAVIS --project_name DAVIS_cold_drug_SnG_concat --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --joint concat'
#            '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_drug_GnS_concat --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --joint concat',
#            '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type graph --d2_type graph --data_name DAVIS --project_name DAVIS_cold_drug_GnG_concat --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --joint concat',
#
#            '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_target_SnS_concat --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --joint concat',
#            '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type seq --d2_type graph --data_name DAVIS --project_name DAVIS_cold_target_SnG_concat --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --joint concat',
#            '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_target_GnS_concat --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --joint concat',
#            '--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type graph --d2_type graph --data_name DAVIS --project_name DAVIS_cold_target_GnG_concat --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --joint concat',
#
#            '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type seq --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_SnS_concat --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --joint concat',
#            '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type seq --d2_type graph --data_name DAVIS --project_name DAVIS_cold_DT_SnG_concat --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --joint concat',
#            '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_DT_GnS_concat --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --joint concat',
#            '--data ./data/preprocessed/DAVIS/cold_split/Drug_and_Target --d1_type graph --d2_type graph --data_name DAVIS --project_name DAVIS_cold_DT_GnG_concat --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --joint concat'
#            ]

import time
for command in cmd_lst:
    start = time.time()
    finish(f'\nstarting {command}\n - server gpu 0')
    subprocess.run(args=[sys.executable, 'CJRM.py'] + split(command))

    f_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    finish(f'\nFinished: {command}\nLearning time: {f_time} - server gpu 0')
finish(f'\nAll job finished successfully!')
