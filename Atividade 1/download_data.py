import kagglehub

# Download latest version
path = kagglehub.dataset_download("yyxian/u-s-airline-traffic-data", output_dir="dataset")

print("Path to dataset files:", path)