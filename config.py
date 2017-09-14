from jobman import DD

RAB_DATASET_BASE_PATH = '/mnt/data3/guoyuyu/datasets/MSVD/predatas/'
RAB_FEATURE_BASE_PATH = '/mnt/data3/guoyuyu/datasets/MSVD/features/Resnet/'
OPT_FEATURE_BASE_PATH = '/mnt/data3/guoyuyu/datasets/MSVD/features/Resnet_ucf101_gray_optflow_tvl_mat_id/'
RAB_EXP_PATH = '/home/guoyuyu/results/youtube/TS_mean_Double_initts_mean_LSTM_NoBN_tem3_ffts2mu_Res/'
'''
RAB_DATASET_BASE_PATH = '/mnt/data3/guoyuyu/datasets/MSR-VTT/predatas/'
RAB_FEATURE_BASE_PATH = '/mnt/data3/guoyuyu/datasets/MSR-VTT/features/Resnet/'
OPT_FEATURE_BASE_PATH = '/mnt/data3/guoyuyu/datasets/MSR-VTT/features/Resnet_ucf101_gray_optflow_tvl_mat/'
ACT_FEATURE_BASE_PATH = '/mnt/data3/guoyuyu/datasets/MSR-VTT/features/TS_LSTM_finetune_res/'
RAB_EXP_PATH = '/home/guoyuyu/results/MSR-VTT/TS_mean_Double_initts_mean_LSTM_NoBN_tem3_ffts2mu_Res/'
'''


config = DD({
    'model': 'attention',
    'random_seed': 1234,
    # ERASE everything under save_model_path
    'erase_history': True,
    'attention': DD({
        'reload_': False,
        'verbose': True,
        'debug': False,
        'save_model_dir': RAB_EXP_PATH + 'save_dir/',
        'from_dir': RAB_EXP_PATH + 'from_dir/',
        # dataset
        'dataset': 'youtube2text', #msr-vtt #youtube2text
        'video_feature': 'googlenet',#res_opt #googlenet
        'K':30, # 26 when compare
        'OutOf':None,
        # network
        'dim_word':512,#468, # 474
        'tu_dim': 512,
        'mu_dim': 512,
        'ctx_dim':-1,# auto set
        'temp_size':3,
        'n_layers_out':1, # for predicting next word
        'n_layers_init':0,
        'encoder_dim': 1024,#300,
        'init_ts':True,
        'get_tslstm':'mean',
        'prev2out':True,
        'ctx2out':True,
        'selector':True,
        'n_words':20000,
        'maxlen':30, # max length of the descprition
        'use_dropout':True,
        'isGlobal': True,
        'dict_type':'small',
        # training
        'patience':20,
        'max_epochs':500,
        'decay_c':1e-4,
        'alpha_entropy_r': 0.,
        'alpha_c':0.70602,
        'lrate':0.0001,
        'optimizer':'adadelta',
        'clip_c': 10.,
        # minibatches
        'batch_size': 64, # for trees use 25
        'valid_batch_size':200,
        'dispFreq':10,
        'validFreq':1000,
        'saveFreq':-1, # this is disabled, now use sampleFreq instead
        'sampleFreq':100,
        # blue, meteor, or both
        'metric': 'everything', # set to perplexity on DVS
        }),
    })
