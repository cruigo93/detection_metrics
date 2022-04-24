install:
	pip3 install -r requirenments.txt
	python3 src/eval_detector.py -g GT.txt -d DT.txt