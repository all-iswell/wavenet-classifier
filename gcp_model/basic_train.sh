#!/bin/bash
TRAIN_DATA=$(pwd)/data/onoff_16000_train.tfrecord
EVAL_DATA=$(pwd)/data/onoff_16000_eval.tfrecord
now=$(date +"%Y%m%d_%H%M%S")
JOB_DIR=$(pwd)/outputs/$now
TRAIN_STEPS=2
LEARNING_RATE=4.0
gcloud ml-engine local train\
	--module-name trainer.task\
	--package-path trainer\
	-- \
	--train-files $TRAIN_DATA \
	--eval-files $EVAL_DATA \
	--job-dir $JOB_DIR \
	--learning-rate $LEARNING_RATE \
	--train-steps $TRAIN_STEPS

