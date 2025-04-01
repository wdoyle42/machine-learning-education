library(jpeg)
library(imager)
library(tibble)
library(stringr)

rasterize_image <- function(image_path, size = 28) {
  img <- load.image(image_path)
  
  # Convert to grayscale if needed

    img <- grayscale(img)
  
  
  # Resize to 28x28
    img_resized <- resize(img, size_x = size, size_y = size)
  
  # Convert to array, drop extra dimensions, flatten
  img_array <- as.array(img_resized)
  img_matrix <- drop(img_array)  # drop all singleton dimensions
  img_matrix<-as.numeric(img_matrix)
  img_matrix <- (1 - img_matrix)
}
# Main function to process images and extract labels
process_directory <- function(directory, output_file = "rasterized_labeled_images.csv") {
  files <- list.files(directory, pattern = "\\.jpe?g$", full.names = TRUE, ignore.case = TRUE)
  
  # Extract label from filename (e.g., Smith_7.jpg → 7)
  extract_label <- function(path) {
    name <- basename(path)
    str_extract(name, "(?<=_)[0-9](?=\\.jpe?g$)")
  }
  
  labels <- sapply(files, extract_label)
  
  
  
  # Process each image
  image_vectors <- lapply(files, rasterize_image)
  
  # Combine into a data frame
  image_df <- as_tibble(do.call(rbind, image_vectors))
  names(image_df) <- paste0("pixel_", seq_len(ncol(image_df)))
  
  # Add label column (as factor or numeric)
  image_df$label <- as.integer(labels)
  
  # Save to CSV
  write.csv(image_df, output_file, row.names = FALSE)
  
  message("Done! Saved labeled data to: ", output_file)
}

process_directory("./",output_file="class_digits.csv")
