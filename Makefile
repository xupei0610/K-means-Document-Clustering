all:
	@echo 'Run "make kmeans", for compiling the K-Means program from source'
	@echo 'For windows platform, a compiled program named sphkmeans.exe is provided'
	@echo 'Run "make process", for generating analyzable data from the reuters-2157878 dataset'

kmeans:
	@echo 'Compiling K-Means Program:'
	g++ -Wall -O3 -std=c++11 -I lib/Eigen/ main.cpp lib/KMeans.cpp -o sphkmeans

preprocess:
	@echo 'Preprocessing Document Data:'
	python3 preprocess.py
