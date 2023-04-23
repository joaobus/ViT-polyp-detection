## Model configs
def get_model_config(get_test_config: bool = False):
    # Input parameters
    H=224
    W=224
    C=3
    P=16
    EMB_SIZE = (P**2)*C
    return_attn_weights = True
    
    # Model parameters    
    if get_test_config:
        DROP_RATE = 0.
        NUM_HEADS = 1
        NUM_BLOCKS = 1
    else:
        DROP_RATE = 0.2
        NUM_HEADS = 8
        NUM_BLOCKS = 6

    config = {'h':H,'w':W,'c':C,'patch_size':P,'emb_size':EMB_SIZE,
              'drop_rate':DROP_RATE,'num_heads':NUM_HEADS,'num_blocks':NUM_BLOCKS,
              'return_attn_weights':return_attn_weights}
    
    return config


## Training configs
def get_training_config(get_test_config: bool = False):
    LR = 1e-4
    THRESHOLD = 0.5
    LR_PATIENCE = 20

    if get_test_config:
        WEIGHT_DECAY = 0.
        N_EPOCHS = 3
    else:
        WEIGHT_DECAY = 1e-9
        N_EPOCHS = 300

    config = {'num_epochs':N_EPOCHS, 'lr':LR, 'weight_decay':WEIGHT_DECAY,
              'threshold':THRESHOLD, 'lr_patience':LR_PATIENCE}

    return config



## Loading configs
def get_loading_config():
    BATCH = 100
    VAL_RATIO = 0.5  # % do dataset de teste
    NUM_WORKERS = 2

    config = {'batch_size':BATCH,'val_ratio':VAL_RATIO,'num_workers':NUM_WORKERS}

    return config

