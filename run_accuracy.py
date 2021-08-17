
import os
import onnx
import glob
import scipy.io
import numpy as np
import logging
from PIL import Image
import onnx
import onnxruntime
import argparse
import time
from ox_utils.calibrate import CalibrationDataReader
from ox_utils.quant_utils import QuantFormat, QuantType
from ox_utils.quantize import quantize_static

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="batch size to calibration and prediction")
    parser.add_argument("--quant_format", type=QuantFormat.from_string, default=QuantFormat.QOperator, choices=list(QuantFormat))
    parser.add_argument("--per_channel", type=bool, default=False, help="whether quantize const operator with per channel")
    parser.add_argument("--fp32_model", required=True, help="fp32 model")
    parser.add_argument("--augmented_model", required=True, help="fp32 model with min/max ")
    parser.add_argument("--int8_model", required=True, help="int8 model")
    parser.add_argument("--dataset", required=True, help="dataset")
    parser.add_argument("--calibration_dataset_size", type=int, default=1, help="size of dataset for calibration, the others are for prediction")

    args = parser.parse_args()
    return args

def split_dataset(dataset_path, calibration_dataset_size, batch_size):
    total_dataset_size = len(os.listdir(dataset_path + "/val"))
    if calibration_dataset_size > total_dataset_size:
        logging.warning(
            "calibration data size is bigger than available dataset. Will assign half of the dataset for calibration")
        calibration_dataset_size = total_dataset_size // 2
    calibration_dataset_size = (calibration_dataset_size // batch_size) * batch_size
    if calibration_dataset_size == 0:
        logging.warning("No dataset is assigned for calibration. Please use bigger dataset")

    prediction_dataset_size = ((total_dataset_size - calibration_dataset_size) // batch_size) * batch_size
    if prediction_dataset_size <= 0:
        logging.warning("No dataset is assigned for evaluation. Please use bigger dataset")
    return [calibration_dataset_size, prediction_dataset_size]

def convert_model_batch_to_dynamic(model_path):
    model = onnx.load(model_path)
    initializers =  [node.name for node in model.graph.initializer]
    inputs = []
    for node in model.graph.input:
        if node.name not in initializers:
            inputs.append(node)
    input_name = inputs[0].name
    shape = inputs[0].type.tensor_type.shape
    dim = shape.dim
    if not dim[0].dim_param:
        dim[0].dim_param = 'N'
        model = onnx.shape_inference.infer_shapes(model)
        model_name = model_path.split(".")
        model_path = model_name[0] + "_dynamic.onnx"
        onnx.save(model, model_path)
    return [model_path, input_name]

def preprocess_images(input, channels=3, height=224, width=224):
    image = input.resize((width, height), Image.ANTIALIAS)
    input_data = np.asarray(image).astype(np.float32)
    if len(input_data.shape) != 2:
        input_data = input_data.transpose([2, 0, 1])
    else:
        input_data = np.stack([input_data] * 3)
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(input_data.shape[0]):
        input_data[channel, :, :] = (input_data[channel, :, :] / 255 - mean[channel]) / std[channel]
    return input_data

class ImageNetDataReader(CalibrationDataReader):
    def __init__(self,
                 image_folder,
                 width=224,
                 height=224,
                 start_index=0,
                 end_index=0,
                 stride=1,
                 batch_size=1,
                 model_path='augmented_model.onnx',
                 input_name='data',
                 providers=['CPUExecutionProvider']):
        '''
        :param image_folder: image dataset folder
        :param width: image width
        :param height: image height 
        :param start_index: start index of images
        :param end_index: end index of images
        :param stride: image size of each data get 
        :param batch_size: batch size of inference
        :param model_path: model name and path
        :param input_name: model input name
        :param providers: cpu/gpu to run
        '''

        self.image_folder = image_folder + "/val"
        self.model_path = model_path
        self.preprocess_flag = True
        self.enum_data_dicts = iter([])
        self.datasize = 0
        self.width = width
        self.height = height
        self.start_index = start_index
        self.end_index = len(os.listdir(self.image_folder)) if end_index == 0 else end_index
        self.stride = stride if stride >= 1 else 1
        self.batch_size = batch_size
        self.input_name = input_name
        self.providers = providers
        self.meta_path = image_folder + "/devkit/data/meta.mat"
        self.synset_words_path = image_folder + "/synset_words.txt"
        self.ground_truth_path = image_folder + "/devkit/data/ILSVRC2012_validation_ground_truth.txt"

    def get_dataset_size(self):
        return len(os.listdir(self.image_folder))

    def get_input_name(self):
        if self.input_name:
            return
        session = onnxruntime.InferenceSession(self.model_path, self.providers)
        self.input_name = session.get_inputs()[0].name

    def get_next(self):
        iter_data = next(self.enum_data_dicts, None)
        if iter_data:
            return iter_data

        self.enum_data_dicts = None
        if self.start_index < self.end_index:
            if self.batch_size == 1:
                data = self.load_serial()
            else:
                data = self.load_batches()

            self.start_index += self.stride
            self.enum_data_dicts = iter(data)

            return next(self.enum_data_dicts, None)
        else:
            return None

    def load_serial(self):
        width = self.width
        height = self.width
        nchw_data_list, filename_list, image_size_list = self.preprocess_imagenet(self.image_folder, height, width,
                                                                                  self.start_index, self.stride)
        input_name = self.input_name

        data = []
        for i in range(len(nchw_data_list)):
            nhwc_data = nchw_data_list[i]
            file_name = filename_list[i]
            data.append({input_name: nhwc_data})
        return data

    def load_batches(self):
        width = self.width
        height = self.height
        batch_size = self.batch_size
        stride = self.stride
        input_name = self.input_name

        batches = []
        for index in range(0, stride, batch_size):
            start_index = self.start_index + index
            nchw_data_list, filename_list, image_size_list = self.preprocess_imagenet(
                self.image_folder, height, width, start_index, batch_size)

            if nchw_data_list.size == 0:
                break

            nchw_data_batch = []
            for i in range(len(nchw_data_list)):
                nhwc_data = np.squeeze(nchw_data_list[i], 0)
                nchw_data_batch.append(nhwc_data)
            batch_data = np.concatenate(np.expand_dims(nchw_data_batch, axis=0), axis=0)
            data = {input_name: batch_data}

            batches.append(data)

        return batches

    def preprocess_imagenet(self, images_folder, height, width, start_index=0, size_limit=0):
        '''
        Loads a batch of images and preprocess them
        parameter images_folder: path to folder storing images
        parameter height: image height in pixels
        parameter width: image width in pixels
        parameter start_index: image index to start with   
        parameter size_limit: number of images to load. Default is 0 which means all images are picked.
        return: list of matrices characterizing multiple images
        '''

        image_names = os.listdir(images_folder)
        image_names.sort()
        if size_limit > 0 and len(image_names) >= size_limit:
            end_index = start_index + size_limit
            if end_index > len(image_names):
                end_index = len(image_names)
            batch_filenames = [image_names[i] for i in range(start_index, end_index)]
        else:
            batch_filenames = image_names

        unconcatenated_batch_data = []
        image_size_list = []

        for image_name in batch_filenames:
            image_filepath = images_folder + '/' + image_name
            img = Image.open(image_filepath)
            if img.mode == 'CMYK':
                img = img.convert('RGB')
            # print("image_path: ", image_filepath)
            image_data = preprocess_images(img)
            image_data = np.expand_dims(image_data, 0)
            unconcatenated_batch_data.append(image_data)
            image_size_list.append(np.array([img.size[1], img.size[0]], dtype=np.float32).reshape(1, 2))

        batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
        return batch_data, batch_filenames, image_size_list

    def get_synset_id(self, image_folder, offset, dataset_size):
        ilsvrc2012_meta = scipy.io.loadmat(self.meta_path)
        id_to_synset = {}
        for i in range(1000):
            id = int(ilsvrc2012_meta["synsets"][i, 0][0][0][0])
            id_to_synset[id] = ilsvrc2012_meta["synsets"][i, 0][1][0]

        synset_to_id = {}
        file = open(self.synset_words_path, "r")
        index = 0
        for line in file:
            parts = line.split(" ")
            synset_to_id[parts[0]] = index
            index = index + 1
        file.close()

        file = open(self.ground_truth_path, "r")
        id = file.read().strip().split("\n")
        id = list(map(int, id))
        file.close()

        image_names = os.listdir(self.image_folder)
        image_names.sort()
        image_names = image_names[offset:offset + dataset_size]
        seq_num = []
        for file in image_names:
            # print("file: ", file)
            seq_num.append(int(file.split("_")[-1].split(".")[0]))
        id = np.array([id[index - 1] for index in seq_num])
        synset_id = np.array([synset_to_id[id_to_synset[index]] for index in id])

        # one-hot encoding
        synset_id_onehot = np.zeros((len(synset_id), 1000), dtype=np.float32)
        for i, id in enumerate(synset_id):
            synset_id_onehot[i, id] = 1.0
        return synset_id_onehot


class ImageClassificationEvaluator:
    def __init__(self,
                 model_path,
                 synset_id,
                 data_reader: CalibrationDataReader):
        '''
        :param model_path: ONNX model to validate
        :param synset_id: ILSVRC2012 synset id        
        :param data_reader: user implemented object to read in and preprocess calibration dataset
                            based on CalibrationDataReader Interface
        '''

        self.model_path = model_path
        self.data_reader = data_reader
        self.prediction_result_list = []
        self.synset_id = synset_id

    def get_result(self):
        return self.prediction_result_list

    def predict(self):
        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 3
        session = onnxruntime.InferenceSession(self.model_path, sess_options=sess_options)

        total = 0.0
        runs = 0
        inference_outputs_list = []
        while True:
            inputs = self.data_reader.get_next()
            if not inputs:
                break
            start = time.perf_counter()
            output = session.run(None, inputs)
            end = (time.perf_counter() - start) * 1000
            total += end
            runs += 1
            inference_outputs_list.append(output)
        total /= runs
        self.prediction_result_list = inference_outputs_list
        return total

    def top_k_accuracy(self, truth, prediction, k=1):
        '''From https://github.com/chainer/chainer/issues/606        
        '''

        y = np.argsort(prediction)[:, -k:]
        return np.any(y.T == truth.argmax(axis=1), axis=0).mean()

    def evaluate(self, prediction_results):
        batch_size = len(prediction_results[0][0])
        total_val_images = len(prediction_results) * batch_size
        y_prediction = np.empty((total_val_images, 1000), dtype=np.float32)
        i = 0
        for res in prediction_results:
            y_prediction[i:i + batch_size, :] = res[0]
            i = i + batch_size
        top1 = self.top_k_accuracy(self.synset_id, y_prediction, k=1)
        top5 = self.top_k_accuracy(self.synset_id, y_prediction, k=5)
        return top1, top5

if __name__ == '__main__':
    '''
    INT8 Inference on Resnet model

    The script is using ILSVRC2012 ImageNet dataset for calibration and prediction.
    Please prepare the dataset as below, 
    1. Create dataset folder 'ILSVRC2012' in workspace.
    2. Download ILSVRC2012 validation dataset and development kit from http://www.image-net.org/challenges/LSVRC/2012/downloads.
    3. Extract validation dataset JPEG files to 'ILSVRC2012/val'.
    4. Extract development kit to 'ILSVRC2012/devkit'. Two files in the development kit are used, 'ILSVRC2012_validation_ground_truth.txt' and 'meta.mat'.
    5. Download 'synset_words.txt' from https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt into 'ILSVRC2012/'.
    
    Please download Resnet50 model from ONNX model zoo https://github.com/onnx/models/blob/master/vision/classification/resnet/model/resnet50-v2-7.tar.gz
    Untar the model into the workspace
    '''

    # Parse config
    args = get_args()
    batch_size = args.batch_size
    quant_format = args.quant_format
    per_channel = args.per_channel
    fp32_model_path = args.fp32_model
    int8_model_path = args.int8_model
    augmented_model_path = args.augmented_model
    dataset_path = args.dataset
    calibration_dataset_size = args.calibration_dataset_size

    # Get calibration and prediction dataset size
    logger.info('Split dataset for calibration and prediction ...')
    [calibration_dataset_size, prediction_dataset_size] = split_dataset(dataset_path,
                                                                        calibration_dataset_size,
                                                                        batch_size)
                        
    # Convert fp32 model static batch to dynamic batch
    logger.info('Convert fp32 model static batch to dynamic batch ...')
    [dynamic_fp32_model_path, input_name] = convert_model_batch_to_dynamic(fp32_model_path)


    # Quantize fp32 model to int8 model
    logger.info('Prepare calibration data reader ...')
    calibration_data_reader = ImageNetDataReader(dataset_path,
                                     start_index=0,
                                     end_index=calibration_dataset_size,
                                     stride=calibration_dataset_size,
                                     batch_size=batch_size,
                                     model_path=augmented_model_path,
                                     input_name=input_name)
    logger.info('Quantize fp32 model to int8 model ...')
    quantize_static(dynamic_fp32_model_path,
                    int8_model_path,
                    calibration_data_reader,
                    quant_format=quant_format,
                    per_channel=per_channel,
                    weight_type=QuantType.QInt8)

    # Prepare prediction data reader for fp32 model
    logger.info('Prepare prediction data reader for fp32 ...')
    fp32_prediction_data_reader = ImageNetDataReader(dataset_path,
                                     start_index=calibration_dataset_size,
                                     end_index=calibration_dataset_size + prediction_dataset_size,
                                     stride=prediction_dataset_size,
                                     batch_size=batch_size,
                                     model_path=dynamic_fp32_model_path,
                                     input_name=input_name)
    logger.info('Generate synset id for fp32 ...')
    fp32_synset_id = fp32_prediction_data_reader.get_synset_id(dataset_path, calibration_dataset_size,
                                          prediction_dataset_size)

    # Run fp32 prediction
    logger.info('Prediction and evaluate fp32 ...')
    fp32_evaluator = ImageClassificationEvaluator(dynamic_fp32_model_path, fp32_synset_id, fp32_prediction_data_reader)
    fp32_runtime = fp32_evaluator.predict()
    fp32_result = fp32_evaluator.get_result()
    fp32_top1, fp32_top5 = fp32_evaluator.evaluate(fp32_result)

    # Prepare prediction data reader for int8 model
    logger.info('Prepare prediction data reader for int8 ...')
    int8_prediction_data_reader = ImageNetDataReader(dataset_path,
                                     start_index=calibration_dataset_size,
                                     end_index=calibration_dataset_size + prediction_dataset_size,
                                     stride=prediction_dataset_size,
                                     batch_size=batch_size,
                                     model_path=int8_model_path,
                                     input_name=input_name)
    logger.info('Generate synset id for int8 ...')
    int8_synset_id = int8_prediction_data_reader.get_synset_id(dataset_path, calibration_dataset_size,
                                          prediction_dataset_size)

    # Run int8 prediction
    logger.info('Prediction and evaluate int8 ...\n')
    int8_evaluator = ImageClassificationEvaluator(int8_model_path, int8_synset_id, int8_prediction_data_reader)
    int8_runtime = int8_evaluator.predict()
    int8_result = int8_evaluator.get_result()
    int8_top1, int8_top5 = int8_evaluator.evaluate(int8_result)

    # Show results
    logger.info('Show results ...')
    logger.info(f"fp32 runtime: {fp32_runtime:.2f}ms")
    logger.info(f"fp32 top 1: {100*fp32_top1:.2f}%")
    logger.info(f"fp32 top 5: {100*fp32_top5:.2f}%\n")

    logger.info(f"int8 runtime: {int8_runtime:.2f}ms")
    logger.info(f"int8 top 1: {100*int8_top1:.2f}%")
    logger.info(f"int8 top 5: {100*int8_top5:.2f}%\n")

    logger.info(f"int8/fp32 performance: {100*fp32_runtime/int8_runtime:.2f}%")
    logger.info(f"int8/fp32 top 1: {100*int8_top1/fp32_top1:.2f}%")
    logger.info(f"int8/fp32 top 5: {100*int8_top5/fp32_top5:.2f}%\n")
