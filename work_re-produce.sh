# train the AKPNet
CUDA_VISIBLE_DEVICES=0 python train_rehearser.py --logs-dir rehearser_pretrain_learn_kernel_c1-g1_mobilenet-v3 --color_style rgb --mobile --n_kernel 1 --groups 1 --learn_kernel
# training order-1
CUDA_VISIBLE_DEVICES=0 python continual_train_extract_proto.py --logs-dir reproduce/setting-1   --mobile 
# training order-2
CUDA_VISIBLE_DEVICES=0 python continual_train_extract_proto.py --logs-dir reproduce/setting-2   --mobile --setting 2
