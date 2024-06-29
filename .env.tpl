[log]
LOG_DATA_PATH='log/data'
LOG_MODEL_PATH='log/model'
LOG_CONFUSION_MATRIX="log/confusion_matrix"

[tokenizer]
TOKENIZER='bert-base-uncased'

[dataversion]
# [MintREC, MintREC2.0]
DATA_VERSION='MintREC'
DATA_DIR='C:/Users/datng/Documents/LAB/KLTN/MIntRec_data/MintREC/'

[hyparameter]
BATCH_SIZE=4
LEARNING_RATE=3e-5
NUM_EPOCH=20

[mult]
MULT_DIM_MODEL=30
MULT_NUM_HEAD=6
MULT_NUM_LAYER=5

[sdif]
SDIF_FEATURE_DIM=768
SDIF_NUMLAYER_SELF_ATTENTION=1
SDIF_NUM_HEAD=8
