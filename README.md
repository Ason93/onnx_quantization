call run.py to calibrate, quantize and run the quantized model, e.g.:

release:
python run.py --input_model models/mobilenet/mobilenetv2-7.onnx --output_model models/mobilenet/mobilenetv2-7.quant.onnx --calibrate_dataset images/test_images/
python run.py --input_model models/resnet/resnet50-v1-13.onnx --output_model models/resnet/resnet50-v1-13.quant.onnx --calibrate_dataset images/test_images

debug:
python -m pdb run.py --input_model models/resnet/resnet50-v1-13.onnx --output_model models/resnet/resnet50-v1-13.quant.onnx --calibrate_dataset images/test_images