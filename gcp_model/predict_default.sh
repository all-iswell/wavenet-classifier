#!/bin/bash

DATA_FORMAT="TF_RECORD"
MODEL_NAME='onoff'
VERSION_NAME='v1'
REGION='us-east1'
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="onoff_batch_predict_$now"
BUCKET_NAME="my-project-1508465090484-mlengine"
INPUT_PATHS="gs://$BUCKET_NAME/data/*"
OUTPUT_PATH="gs://$BUCKET_NAME/$JOB_NAME"
MAX_WORKER_COUNT="20" 


gcloud ml-engine jobs submit prediction $JOB_NAME\
	--model $MODEL_NAME\
	--input-paths $INPUT_PATHS\
	--output-path $OUTPUT_PATH\
	--region $REGION\
	--data-format $DATA_FORMAT
