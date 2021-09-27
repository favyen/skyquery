SkyQuery
--------

SkyQuery is a proof-of-concept platform for applications involving aerial drone
video sensing, such as traffic monitoring, infrastructure inspection, and
wildlife population management.

Project webpage: https://favyen.com/skyquery/


Setup
-----

First, download the SkyQuery dataset. In the commands below, we will assume it
is in /data/:

	wget https://favyen.com/files/skyquery-dataset.zip
	unzip skyquery-dataset.zip
	mv data/ /data/

Install Python dependencies:

	pip install -r requirements.txt

Also setup YOLOv3 and TensorFlow, which are needed to run the car detector and
pedestrian detector, respectively:

	cd /path/to/skyquery3
	git clone https://github.com/pjreddie/darknet
	cd darknet
	make


Detector Registration
---------------------

We now need to register object detection models with the SkyQuery web platform,
which we will run in a later step. This will save metadata about the models to
/data/detect/X.json.

	python detector/register.py pedestrians /data/data/ /data/pedestrian-model/model 1
	python yolov3/register.py cars /data/data/ /data/car-model/yolov3.cfg /data/car-model/yolov3.best


Frame Alignment
---------------

Run the frame alignment script. This will input video from /data/frames/main/
along with GPS data from /data/data/align-gps.json, and produce a file
describing the bounds of each frame in /data/data/align-out.json.

	python preprocess_fast/main.py /data/frames/main/ /data/data/ 2


Web Platform
------------

Now we can run the web platform. Install Go if needed:

	sudo apt install golang-go

And then (note: we use Go 1.13, for newer version you may need to disable Go modules):

	go get github.com/mitroadmaps/gomapinfer/common
	go run ./web/ /data/data/ /data/frames/main/

You can now run the example programs in programs/ folder using the web
interface (http://localhost:8080/).
