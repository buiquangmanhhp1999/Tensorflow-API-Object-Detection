# Tensorflow-API-Object-Detection
Tensorflow API SSD MobileNet V2 Object Detection

### 1. Install Library
```
pip install -r requirement.txt
pip install -U cython
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

### 2. Split Dataset to train, test 
```python
# usage: partition_dataset.py [-h] [-i IMAGEDIR] [-o OUTPUTDIR] [-r RATIO] [-x]
python ./scripts/partion_dataset.py -i ./scripts/Dataset/ -x -r 0.1
```
### 3. Create Label Map
```
item {
  id: 1
  name: 'id'
}
```
### 4. Create TensorFlow Records
First, run command line
```
protoc object_detection/protos/*.proto --python_out=.
```
Then, run above command to create tf train record and test record
```python
# usage: create_tf_record.py [-h] [-x XML_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]

# create train record
python ./scripts/create_tf_record.py -x ./scripts/TrainValDataset/train/ -l ./scripts/label_map.pbxt -o ./scripts/TrainValDataset/train.record

# create test record
python ./scripts/create_tf_record.py -x ./scripts/TrainValDataset/test/ -l ./scripts/label_map.pbtxt -o ./scripts/TrainValDataset/test.record

```
### 5. Train
Run above command to train model
```
python ./object_detection/model_main_tf2.py --model_dir=./ssd_mobilenet_v2/ --pipeline_config_path=./ssd_mobilenet_v2/pipeline.config 
```
