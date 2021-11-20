# Run example
call run.py to calibrate, quantize and run the quantized model, e.g.:

## release:
```python
python run_benchmark.py --input_model /Users/chengxiongjin/Downloads/Models/ONNX/resnet50_v1/resnet50-v1-13.onnx --output_model /Users/chengxiongjin/Downloads/Models/ONNX/resnet50_v1/resnet50-v1-13.quant.onnx --augmented_model /Users/chengxiongjin/Downloads/Models/ONNX/resnet50_v1/resnet50-v1-13.augmented.onnx  --calibrate_dataset images/test_images
```
```python
python run_accuracy.py --batch_size 1 --fp32_model /Users/chengxiongjin/Downloads/Models/ONNX/resnet50_v1/resnet50-v1-13.onnx --int8_model /Users/chengxiongjin/Downloads/Models/ONNX/resnet50_v1/resnet50-v1-13.quant.onnx --dataset /Users/chengxiongjin/Downloads/Images/ILSVRC2012_100 --calibration_dataset_size 50
```
## debug:
```python
python -m pdb run_benchmark.py --input_model /Users/chengxiongjin/Downloads/Models/ONNX/resnet50_v1/resnet50-v1-13.onnx --output_model /Users/chengxiongjin/Downloads/Models/ONNX/resnet50_v1/resnet50-v1-13.quant.onnx --augmented_model /Users/chengxiongjin/Downloads/Models/ONNX/resnet50_v1/resnet50-v1-13.augmented.onnx  --calibrate_dataset images/test_images
```
```python
python -m pdb run_accuracy.py --batch_size 1 --fp32_model /Users/chengxiongjin/Downloads/Models/ONNX/resnet50_v1/resnet50-v1-13.onnx --int8_model /Users/chengxiongjin/Downloads/Models/ONNX/resnet50_v1/resnet50-v1-13.quant.onnx --dataset /Users/chengxiongjin/Downloads/Images/ILSVRC2012_100 --calibration_dataset_size 50
```

## run visiualize distribution
```python
python run_vis_dist.py --input_path=resnet_v1_50_output.npy --output_path=resnet_v1_50_output_dist.png
```