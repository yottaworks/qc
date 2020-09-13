import boto3
import configparser
import cv2 as cv
import getopt
import logging
import os
import queue
import sys
import torch
import torch.onnx
import uuid
from os import path
from threading import Thread
from torch.autograd import Variable
from torchvision import transforms

LOG_FORMAT = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"

logging.basicConfig(filename='build_model.log', level=logging.DEBUG, format=LOG_FORMAT)
logFormatter = logging.Formatter(LOG_FORMAT)
rootLogger = logging.getLogger()

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)

rootLogger.addHandler(consoleHandler)


def usage():
    print("Usage: QuayChain -m <ONNX recognition model> -b <s3 bucket name> -r <aws region>")


processing_queue = queue.Queue()


def capture_frame(vc, q):
    while True:
        return_value, frame = vc.read()

        if return_value:
            logging.info('Read frame from RTSP stream: %s', return_value)
            q.put(frame)

        if not q.empty():
            logging.info('Joining queue.')
            q.join()


def process_frame(image_tensor, device, model):
    labels = ['Not a shipping container', 'shipping container', 'shipping container front']
    #       image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    predict_input = Variable(image_tensor)
    predict_input = predict_input.to(device)
    output = model(predict_input)

    output = torch.exp(output)
    probabilities, classes = output.topk(1, dim=1)

    message = 'Model is {:01.1f}% certain image is {}'.format(probabilities.item() * 100, labels[classes.item()])

    logging.info(message)

    return classes.item()


def upload_to_aws(image, region, bucket):
    try:
        file_name = "test-{}.jpg".format(str(uuid.uuid1()))
        image_file = image.save(file_name)
        with open(file_name, 'rb') as f:
            client = boto3.client('s3', region_name=region)
            client.upload_fileobj(f, bucket, file_name)

        os.remove(file_name)
    except Exception as e:
        logging.error("Failed to upload image to AWS 3: {}".format(e))


def main():
    config = configparser.ConfigParser()
    config.read('qc.config')

    try:
        # options, args = getopt.getopt(sys.argv[1:], "m:b:r:")
        options, args = getopt.getopt(sys.argv[1:], "e")
    except getopt.GetoptError as err:
        # print help information and exit:
        print(str(err))  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    for o, a in options:
        if o == '-m':
            model_file = a
        elif o == '-b':
            bucket = a
        elif o == '-r':
            region = a
        elif o == '-e':
            environment = a
        else:
            usage()
            sys.exit(1)

    # model_file = None
    # bucket = None
    # region = None
    # config_file = 'qc.config'
    environment = 'DEFAULT'

    model_file = config[environment]['model']
    bucket = config[environment]['bucket']
    region = config[environment]['region']
    rtsp_url = config[environment]['rtsp_url']

    if model_file is None or region is None or bucket is None:
        usage()
        sys.exit(2)

    logging.info('Using model %s', model_file)

    if not path.exists(model_file):
        logging.error("Model file does not exist: %s", model_file)
        sys.exit(3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_file)
    model.eval()

    # Make sure the transform matches what is used to train the model
    test_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    to_pil = transforms.ToPILImage()

    logging.info("Begin ")

    video_capture = cv.VideoCapture("rtsp://10.0.0.80/cam1/mpeg4")
    video_capture = cv.VideoCapture(rtsp_url)

    #    while True:
    #        return_value, frame = video_capture.read()

    #        if return_value:
    #            cv.resizeWindow('QuayChain', 640, 480)
    #            cv.imshow('QuayChain', cv.resize(frame, (640, 480)))
    #        else:
    # reconnect to the video stream

    stream_processor = Thread(target=capture_frame, args=(video_capture, processing_queue))
    stream_processor.setDaemon(True)
    stream_processor.start()

    while True:
        try:
            #   logging.debug('Queue size=%d', processing_queue.qsize())
            frame = processing_queue.get(block=True, timeout=1.0)

            if frame is not None:
                logging.debug('Displaying frame.')
                cv.resizeWindow('QuayChain', 640, 480)
                cv.imshow('QuayChain', cv.resize(frame, (640, 480)))
                image = to_pil(frame)
                classification = process_frame(test_transforms(image).float(), device, model)
                if classification == 2 or classification == 1:
                    logging.info('Found container')
                    upload_to_aws(image, region, bucket)
                    logging.info('Upload to S3')
                    break
                processing_queue.task_done()

        except queue.Empty:
            logging.debug("Timed out waiting for a frame.")

        if cv.waitKey(20) & 0xff == ord('q'):
            break

    logging.info("Complete")


if __name__ == '__main__':
    main()
