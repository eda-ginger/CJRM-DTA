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


cuda_num = 0
cmd_lst = [f"--data ./data/preprocessed/DAVIS/random --d1_type graph --d2_type seq --data_name DAVIS --project_name GraphDTA_random_MHA-Cross_22-0 --virtual_node --joint cross_att --use_cuda {cuda_num}"]

import time
for command in cmd_lst:
    start = time.time()
    finish(f'\nstarting {command}\n - server 22-0')
    subprocess.run(args=[sys.executable, 'CJRM.py'] + split(command))

    f_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    finish(f'\nFinished: {command}\nLearning time: {f_time} - server 22-0')
finish(f'\nAll job finished successfully!')
