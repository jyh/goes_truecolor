#!/bin/bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/jyh/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/jyh/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/jyh/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/jyh/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# <<< conda initialize <<<

conda activate create_cloud_masks

cd "${HOME}/projects/goes_truecolor"

/usr/bin/flock -n make_cloud_masks.lock make run_cloud_masks_local
