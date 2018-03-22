#!/bin/bash
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine
JOB_NAME=$1
RESTORE_FROM=$2
REGION=us-central1
TRAIN_DATA=gs://$BUCKET_NAME/data/onoff_16000_train.tfrecord
EVAL_DATA=gs://$BUCKET_NAME/data/onoff_16000_eval.tfrecord
TRAIN_STEPS=3110
EVAL_FREQUENCY=10
echo $BUCKET_NAME

gcloud ml-engine jobs submit training $JOB_NAME \
        --job-dir gs://$BUCKET_NAME/$JOB_NAME \
	--config config.yaml \
	--runtime-version 1.4 \
	--module-name trainer.task \
	--package-path trainer/ \
	--region $REGION \
	-- \
	--train-files $TRAIN_DATA \
	--eval-files $EVAL_DATA \
	--train-steps $TRAIN_STEPS \
	--eval-frequency $EVAL_FREQUENCY \
	--restore-from gs://$BUCKET_NAME/$RESTORE_FROM \
	--verbosity DEBUG

