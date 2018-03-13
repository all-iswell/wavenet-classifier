#!/bin/bash
TRAIN_DATA=$(pwd)/data/onoff_16000_train.tfrecord
EVAL_DATA=$(pwd)/data/onoff_16000_eval.tfrecord
TRAIN_STEPS=2
gcloud ml-engine local train\
	--module-name trainer.task\
	--package-path trainer\
	-- \
	--train-files $TRAIN_DATA \
	--eval-files $EVAL_DATA \
	--train-steps $TRAIN_STEPS

