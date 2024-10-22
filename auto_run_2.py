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


cmd_lst = ['--data ./data/preprocessed/DAVIS/cold_split/Target --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_target_GnS_ban --joint bi_att --virtual_node --n_workers 4 --batch_size 128 --one_shot'
           ]

# '--data ./data/preprocessed/KIBA/random --d1_type graph --d2_type seq --data_name KIBA --project_name KIBA_random_GnS_ban --joint bi_att --one_shot --virtual_node --n_workers 4 --batch_size 128 --use_cuda 1',

import time
for command in cmd_lst:
    start = time.time()
    finish(f'\nstarting {command}\n - server gpu 1')
    subprocess.run(args=[sys.executable, 'CJRM.py'] + split(command))

    f_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    finish(f'\nFinished: {command}\nLearning time: {f_time} - server gpu 1')
finish(f'\nAll job finished successfully!')

#### done

### DAVIS

## concat
# '--data ./data/preprocessed/KIBA/random --d1_type graph --d2_type seq --data_name KIBA --project_name KIBA_random_GnS_concat --use_cuda 1 --joint concat --one_shot'

# '--data ./data/preprocessed/KIBA/random --d1_type seq --d2_type seq --data_name KIBA --project_name KIBA_random_SnS_concat --use_cuda 1 --joint concat --one_shot',
# '--data ./data/preprocessed/KIBA/random --d1_type seq --d2_type seq --data_name KIBA --project_name KIBA_random_SnS_multiple --use_cuda 1 --joint multiple --one_shot',
# '--data ./data/preprocessed/KIBA/random --d1_type seq --d2_type seq --data_name KIBA --project_name KIBA_random_SnS_bi --use_cuda 1 --joint bi --one_shot',
#
# '--data ./data/preprocessed/KIBA/random --d1_type graph --d2_type seq --data_name KIBA --project_name KIBA_random_GnS_multiple --use_cuda 1 --joint multiple --one_shot',
# '--data ./data/preprocessed/KIBA/random --d1_type graph --d2_type seq --data_name KIBA --project_name KIBA_random_GnS_bi --use_cuda 1 --joint bi --one_shot',
#
# '--data ./data/preprocessed/KIBA/random --d1_type seq --d2_type seq --data_name KIBA --project_name KIBA_random_SnS_add --use_cuda 1 --joint add --one_shot',
# '--data ./data/preprocessed/KIBA/random --d1_type graph --d2_type seq --data_name KIBA --project_name KIBA_random_GnS_add --use_cuda 1 --joint add --one_shot'


# '--data ./data/preprocessed/DAVIS/random --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_random_GnS_ban --joint bi_att --use_cuda 1 --virtual_node',
# '--data ./data/preprocessed/DAVIS/cold_split/Drug --d1_type graph --d2_type seq --data_name DAVIS --project_name DAVIS_cold_drug_GnS_ban --joint bi_att --use_cuda 1 --virtual_node',