# Tensorflow-API-Object-Detection
TensorflowV2  API Object Detection

### 1. Split Dataset to train, test 
```python
# usage: partition_dataset.py [-h] [-i IMAGEDIR] [-o OUTPUTDIR] [-r RATIO] [-x]
python ./scripts/partion_dataset.py -i ./scripts/Dataset/ -x -r 0.1
```
### 2. Create Label Map
```
item {
  id: 1
  name: 'id'
}
```
### 3. Create TensorFlow Records
First, run command line
```
protoc object_detection/protos/*.proto --python_out=.
```
Then, run
```python
# usage: create_tf_record.py [-h] [-x XML_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]
python ./scripts/create_tf_record.py -x ./scripts/TrainValDataset/train/ -l ./scripts/label_map.pbxt -o ./scripts/TrainValDataset/train.record
```
