# 리드미
batch_size = hyperparams["data"]["BATCH_SIZE"]
image_size = hyperparams["data"]["IMAGE_SIZE"]
crops_0pct_name = hyperparams["data"]["CROPS_0PCT"]["CROPS_0PCT_NAME"]
crops_0pct_split_dir = Path(hyperparams["data"]["CROPS_0PCT"]["DATASET_DIR"])
crops_10pct_name = hyperparams["data"]["CROPS_10PCT"]["CROPS_10PCT_NAME"]
crops_10pct_split_dir = Path(hyperparams["data"]["CROPS_10PCT"]["DATASET_DIR"])
crops_25pct_name = hyperparams["data"]["CROPS_25PCT"]["CROPS_25PCT_NAME"]
crops_25pct_split_dir = Path(hyperparams["data"]["CROPS_25PCT"]["DATASET_DIR"])

vits16_name = hyperparams["model"]["vits16"]["NAME"]
vits16_path = Path(hyperparams["model"]["vits16"]["PATH"])
vitb16_name = hyperparams["model"]["vitb16"]["NAME"]
vitb16_path = Path(hyperparams["model"]["vitb16"]["PATH"])
vitl16_name = hyperparams["model"]["vitl16"]["NAME"]
vitl16_path = Path(hyperparams["model"]["vitl16"]["PATH"])
vith16plus_name = hyperparams["model"]["vith16plus"]["NAME"]
vith16plus_path = Path(hyperparams["model"]["vith16plus"]["PATH"])

num_classes = hyperparams["model"]["NUM_CLASSES"]
lora_rank = hyperparams["model"]["LORA_RANK"]
lora_alpha = hyperparams["model"]["LORA_ALPHA"]
target_modules = hyperparams["model"]["TARGET_MODULES"]

num_epochs = hyperparams["train"]["NUM_EPOCHS"]
learning_rate = float(hyperparams["train"]["LEARNING_RATE"])
weight_decay = float(hyperparams["train"]["WEIGHT_DECAY"])
early_stopping_patience = hyperparams["train"]["EARLY_STOPPING_PATIENCE"]
checkpoint_dir = Path(hyperparams["train"]["CHECKPOINT_DIR"])