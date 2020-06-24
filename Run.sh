#! /bin/bash

python train.py --lr 0.005 --momentum 0.9 --num_hidden 3 --sizes 100,100,100 --activation tanh --loss ce --opt adam --batch_size 50 --epochs 30 --anneal True --save_dir "/Users/apandey6/Documents/DeeplLearningNPTEL/Assignment_1/model/" --expt_dir "/Users/apandey6/Documents/DeeplLearningNPTEL/Assignment_1/model/Logs/"  --train "/Users/apandey6/Documents/DeeplLearningNPTEL/Assignment_1/Data/train.csv" --val "/Users/apandey6/Documents/DeeplLearningNPTEL/Assignment_1/Data/valid.csv" --test "/Users/apandey6/Documents/DeeplLearningNPTEL/Assignment_1/Data/test.csv"
