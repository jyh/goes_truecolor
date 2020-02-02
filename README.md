# truecolor

Truecolor satellite imagery

To reproduce, follow these steps:

1. Open CloudShell following the directions in https://cloud.google.com/shell/docs/quickstart -- this gives you a free 
temporary VM. If you want something more permanent, start a Compute Engine instance or install the gcloud SDK to your laptop following the directions in https://cloud.google.com/sdk/install

2. Install the necessary Python packages:
```
sudo pip install -U absl-py dateparser netcdf4 xarray pyresample google-cloud-storage apache-beam[gcp] skikit-image tensorflow
```

3. Git clone this repository and cd into it:
```
git clone https://github.com/jyh/truecolor.git
cd truecolor
```

4. Create a bucket to store your outputs if necessary following steps in https://cloud.google.com/storage/docs/creating-buckets

5. Launch a Dataflow job to walk through the satellite data and create TensorFlow records (change the BUCKET name to match yours):
```
BUCKET=weather-tmp
PROJECT_ID=weather-324
OUTDIR=gs://${BUCKET}/truecolor
gsutil -m rm -rf ${OUTDIR}

python -m truecolor.precproc.make_truecolor_examples \
   --out_dir=${OUTDIR} \
   --start_time="1/1/2019 --end_time="1/31/2019" \
   --runner=DataflowRunner
```
This will take about 1.5 hours and cost about $17.

6. Next, launch ML training job on Cloud ML Engine:
```
BUCKET=wrewhoeirwhwoierhowihetwheio
LATLONRES=0.02
TRAIN_RADIUS=32
LABEL_RADIUS=1
DATADIR=gs://${BUCKET}/lightning/preproc_${LATLONRES}_${TRAIN_RADIUS}_${LABEL_RADIUS}/tfrecord

#for ARCH in feateng convnet dnn resnet; do
for ARCH in convnet; do
  JOBNAME=ltgpred_${ARCH}_$(date -u +%y%m%d_%H%M%S)
  OUTDIR=gs://${BUCKET}/lightning/${ARCH}_trained_gpu
  gsutil -m rm -rf $OUTDIR
  gcloud ml-engine jobs submit training $JOBNAME \
      --module-name trainer.train_cnn --package-path ${PWD}/ltgpred/trainer --job-dir=$OUTDIR \
      --region=${REGION} --scale-tier CUSTOM --config largemachine.yaml \
      --python-version 3.5 --runtime-version 1.10 \
      -- \
      --train_data_path ${DATADIR}/train-* --eval_data_path ${DATADIR}/eval-* \
      --train_steps 5000 --train_batch_size 256 --num_cores 4 --arch $ARCH \
      --num_eval_records 1024000 --nlayers 5 --dprob 0 --ksize 3 --nfil 10 --learning_rate 0.01 
done
```
This will take about 30 minutes and cost about $5.
