CUDA_VISIBLE_DEVICES=0 python3 train.py \
	--algorithm svea \
	--seed 0 \
	--domain_name cartpole \
	--task_name swingup \
	--exp_name cartpole_colour_correlation_svea \
	--num_train_steps 250000 \
	--num_test_steps 62500 \
	--action_repeat 8 \
	--correlated_with_colour True
