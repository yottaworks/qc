import boto3
import configparser
import cv2 as cv
import getopt
import logging
import os
import signal
import sys
import _thread
import torch
import torch.onnx
import uuid
from logging.handlers import RotatingFileHandler
from os import path
from time import sleep
from torch.autograd import Variable
from torchvision import transforms

LOG_FORMAT = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
CONFIG_FILE = "qc.config"

logging.basicConfig(filename='qc.log', level=logging.DEBUG, format=LOG_FORMAT)
logFormatter = logging.Formatter(LOG_FORMAT)
rootLogger = logging.getLogger()

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)

rootLogger.addHandler(consoleHandler)

rotatingFileHandler = RotatingFileHandler("/var/log/quaychain.log", maxBytes=1000000, backupCount=5)
rotatingFileHandler.setFormatter(logFormatter)
rootLogger.addHandler(rotatingFileHandler)

is_scoring = False
is_running = False

CONTAINER = 0
NO_CONTAINER = 1
state = NO_CONTAINER


def usage():
    print("Usage: QuayChain -e <optional configuration>")


def capture_frame(vc, q):
    while True:
        return_value, frame = vc.read()

        if return_value:
            logging.info('Read frame from RTSP stream: %s', return_value)
            q.put(frame)

        if not q.empty():
            logging.info('Joining queue.')
            q.join()


def process_frame(image, tensor_transforms, device, model, region, bucket):
    global is_scoring

    is_scoring = True
    labels = ['Not a shipping container', 'shipping container', 'shipping container front']
    image_tensor = tensor_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    predict_input = Variable(image_tensor)
    predict_input = predict_input.to(device)
    output = model(predict_input)

    output = torch.exp(output)
    probabilities, classes = output.topk(1, dim=1)

    message = 'Model is {:01.1f}% certain image is {}'.format(probabilities.item() * 100, labels[classes.item()])
    logging.info(message)

    global state

    if state == NO_CONTAINER:
        if classes.item() == 1 or classes.item() == 2:
            upload_to_aws(image, region, bucket)
            logging.info("New container found")
            state = CONTAINER
    else:
        if classes.item() == 0:
            logging.info("Container clear of view")
            state = NO_CONTAINER

    is_scoring = False

    return classes.item()


def upload_to_aws(image, region, bucket):
    try:
        file_name = "test-{}.jpg".format(str(uuid.uuid1()))
        image.save(file_name)
        with open(file_name, 'rb') as f:
            client = boto3.client('s3', region_name=region)
            client.upload_fileobj(f, bucket, file_name)

        os.remove(file_name)
    except Exception as e:
        logging.error("Failed to upload image to AWS 3: {}".format(e))


def receive_signal(signal_number, frame):
    global is_running
    logging.info("Received signal {}".format(signal_number))
    is_running = False


def main():

    if not os.path.exists(CONFIG_FILE):
        logging.error("Missing configuration file")
        sys.exit(4)

    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    try:
        options, args = getopt.getopt(sys.argv[1:], "e")
    except getopt.GetoptError as err:
        # print help information and exit:
        print(str(err))  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    environment = 'DEFAULT'
    for o, a in options:
        if o == '-e':
            environment = a
        else:
            usage()
            sys.exit(1)
    try:
        model_file = config[environment]['model']
        bucket = config[environment]['bucket']
        region = config[environment]['region']
        rtsp_url = config[environment]['rtsp_url']
    except KeyError:
        logging.error("Bad configuration file (missing key)")
        sys.exit(6)

    if model_file is None or region is None or bucket is None or rtsp_url is None:
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

    logging.info("Starting")

    global is_scoring
    global is_running

    # register the signals to be caught
    signal.signal(signal.SIGHUP, receive_signal)
    signal.signal(signal.SIGINT, receive_signal)
    signal.signal(signal.SIGQUIT, receive_signal)
    signal.signal(signal.SIGILL, receive_signal)
    signal.signal(signal.SIGTRAP, receive_signal)
    signal.signal(signal.SIGABRT, receive_signal)
    signal.signal(signal.SIGBUS, receive_signal)
    signal.signal(signal.SIGFPE, receive_signal)
    #signal.signal(signal.SIGKILL, receive_signal)
    signal.signal(signal.SIGUSR1, receive_signal)
    signal.signal(signal.SIGSEGV, receive_signal)
    signal.signal(signal.SIGUSR2, receive_signal)
    signal.signal(signal.SIGPIPE, receive_signal)
    signal.signal(signal.SIGALRM, receive_signal)
    signal.signal(signal.SIGTERM, receive_signal)

    is_running = True
    try:
        video = cv.VideoCapture(rtsp_url)
        while is_running:
            ret, frame = video.read()
            if ret and is_scoring is False:
                logging.debug('Processing frame')
                image = to_pil(frame)
                _thread.start_new_thread(process_frame, (image, test_transforms, device, model, region, bucket))
                sleep(0.25)
            elif ret is False:
                logging.info("Re-connecting")
                video.release()
                video = cv.VideoCapture(rtsp_url)
    except KeyboardInterrupt:
        logging.error("Keyboard interrupted - stopping now")
    except:
        e = sys.exc_info()[0]
        print("Unexpected error with prediction: %s" % e)

    logging.info("Shutdown complete")


if __name__ == '__main__':
    main()
