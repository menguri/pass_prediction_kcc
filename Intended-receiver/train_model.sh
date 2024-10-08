python train.py \
--trial 48 \
--model transformer \
--n_features 6 \
--n_classes 11 \
--context_dim 32 \
--trans_dim 256 \
--dropout 0 \
--n_epochs 500 \
--start_lr 1e-4 \
--min_lr 1e-5 \
--batch_size 32 \
--print_every_batch 50 \
--save_every_epoch 10 \
--seed 1 \
--cuda \
--add_info

python receiver_train.py \
--trial 26 \
--model transformer \
--n_features 6 \
--n_classes 11 \
--context_dim 32 \
--trans_dim 256 \
--dropout 0 \
--n_epochs 500 \
--start_lr 1e-4 \
--min_lr 1e-5 \
--batch_size 32 \
--print_every_batch 50 \
--save_every_epoch 10 \
--seed 1 \
--cuda 

python pretrain.py \
--trial 50 \
--model transformer \
--n_features 6 \
--n_classes 11 \
--context_dim 32 \
--trans_dim 256 \
--dropout 0.0 \
--n_epochs 100 \
--start_lr 1e-4 \
--min_lr 1e-5 \
--batch_size 32 \
--print_every_batch 50 \
--save_every_epoch 10 \
--seed 1 \
--cuda \
--add_info \
--pretrain 47
