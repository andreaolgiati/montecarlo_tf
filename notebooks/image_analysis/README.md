# Example of Image Classification Distribution Analysis for Montecarlo

## Data Collection
Run the following command to download a set of images into `out/`
```
sh> mkdir out
sh> python download_images.py
```

## Data Preparation
This command creates a `.json` file that has the same format as the MC capture. It will be limited to the first 100 images
```
sh> find out/ |  head -100 | xargs python mkimages.py > capture_small.json
```

## Notebook
To run the notebook:
```
sh> jupyter-notebook MontecarloAnalysis.ipynb
```
