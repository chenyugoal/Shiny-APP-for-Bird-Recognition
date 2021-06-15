# Import libraries
# (remember to set python interpreter, GPU is recommended)
library(keras)
library(EBImage)
library(tensorflow)

# Functions
show_image <- function(path){
  
  if (!is.character(path)){
    stop("The file path must be a character string")
  }
  
  image <- imager::load.image(path)
  plot(image)
}
get_image_info <- function(img){
  image_result <- list(img_width = imager::width(img),
                       img_height = imager::height(img),
                       img_depth = imager::depth(img),
                       img_colour_channels=imager::spectrum(img))
  
  return(image_result)
  
}
index.top.N = function(xs, N=10){
  if(length(xs) > 0) {
    o = order(xs, na.last=FALSE)
    o.length = length(o)
    if (N > o.length) N = o.length
    rev(o[((o.length-N+1):o.length)])
  }
  else {
    0
  }
}
# Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset
# https://www.kaggle.com/veeralakrishna/200-bird-species-with-11788-images
# 
loc_dataset <- "C:/Users/Administrator/Desktop/CUB-200-2011/CUB_200_2011"
loc_images <- file.path(loc_dataset,'images')
loc_classes <- "./classes.txt" 

loc_history <- "F:/Programming/R/batch_history"

# Prepare the class list
# optional list of class subdirectories (e.g. c('dogs', 'cats')). 
classes <- read.csv(loc_classes, header=FALSE, sep="")
classes <- classes[[2]] %>% strsplit(split = '[.]')

clas = c()
for (i in classes){
  clas = c(clas,i[[2]])
}

# Define the Batch Generator
datagen <- image_data_generator(
  rotation_range = 30,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  fill_mode = "nearest",
  horizontal_flip = TRUE,
  vertical_flip = TRUE,
  validation_split = 0.2)

train_generator <- flow_images_from_directory(
  loc_images,
  datagen,
  target_size = c(224, 224),
  batch_size = 64,
  seed = 1,
  save_to_dir = loc_history,
  subset = "training"
)

val_generator <- flow_images_from_directory(
  loc_images,
  datagen,
  target_size = c(224, 224),
  batch_size = 64,
  seed = 1,
  subset = "validation"
)

# Model with RESNET50
pretrained <- application_resnet50(weights = 'imagenet',
                                   include_top = FALSE,
                                   input_shape = c(224, 224, 3))
#freeze_weights(pretrained,from = "conv1_conv", to = "conv4_block6_out")
model <- keras_model_sequential() %>% 
  pretrained %>%
  layer_global_average_pooling_2d() %>%
  layer_flatten() %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dense(units = 200, activation = 'softmax')
freeze_weights(pretrained)
summary(model)
# Compile
model %>% compile(loss = "categorical_crossentropy",
                  optimizer = 'adam',
                  metrics = 'accuracy')

# Fit model
history <- model %>% fit(
  train_generator,
  steps_per_epoch = 147,
  epochs = 10,
  validation_data = val_generator
)

# Save model
model %>% save_model_hdf5("./model/birds_classification.h5")

# Prediction
new_model <- load_model_hdf5("./model/birds_classification.h5")
summary(new_model)

img <- image_load("C:/Users/Administrator/Desktop/311635911.jpg", target_size = c(224,224))
x <- image_to_array(img)
x <- array_reshape(x, c(1, dim(x)))
preds <- new_model %>% predict(x)
ind <- index.top.N(preds, N=5)


# Create the data frame.
# 
df <- data.frame(
  Object = class[ind], 
  Score = preds[ind]
)
# Print the data frame.			
print(df) 
