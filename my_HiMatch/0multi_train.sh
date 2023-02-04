CUDA_VISIBLE_DEVICES=0 python main.py --tag=0fold_himatch_modified_sampling --fold=0 --config_path='config/0settings.json' \
                                      --weight_decay=1e-3 \
                                      --server='lyceum4' --logging=True 