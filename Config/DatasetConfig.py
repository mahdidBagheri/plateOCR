plate_format = "$$###$$"
blank_num = 0
image_width = 512
image_height = 128
batch_size = 8
prediction_head_num = 16
prediction_length = len(plate_format) + blank_num
alphabet_length = 36
train_dataset_size = 800
test_dataset_size = 80