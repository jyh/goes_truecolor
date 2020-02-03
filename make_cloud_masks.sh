#!/bin/bash

set -e

export PATH="/home/jyh/anaconda3/bin:$PATH"
export GOOGLE_APPLICATION_CREDENTIALS=/home/jyh/.credentials/weather-324-586457f473f8.json

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

source activate create_cloud_masks

cd "${HOME}/projects/goes_truecolor"

START_DATE="$(date --utc --date="7 days ago")" 
END_DATE="$(date --utc --date=tomorrow)"

/usr/bin/flock -n make_cloud_masks.lock \
	python -m goes_truecolor.beam.make_cloud_masks \
	--start_date="${START_DATE}" \
	--end_date="${END_DATE}"
