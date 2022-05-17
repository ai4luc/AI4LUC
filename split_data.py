import splitfolders

### Split data into Train and testing

input_folder = "/Users/mateus.miranda/INPE-CAP/mestrado/LUCIai project/algorithm/cerraNet_v3/data/sets/dataset1/"
output_folder = "/Users/mateus.miranda/INPE-CAP/mestrado/LUCIai project/algorithm/cerraNet_v3/data/dataset_cerradov3_NIR+G+B_splited_50k/"

# ratio of split are in order of train/val/test.
splitfolders.ratio(input_folder, output_folder, seed=42, ratio=(.8, .1, .1))

