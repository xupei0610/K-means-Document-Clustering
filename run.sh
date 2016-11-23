#!/bin/sh

#
#  KMeans.hpp
#  Automatic test runing script for K-means Clustering
#
#  @author Pei Xu
#  @version 1.0 11/21/2016
#


echo "This is an autoscript to run a group tests for the K-means coded by Pei Xu."
echo "It will test the following combaination of parameters:"
echo "  Data Model: bag-of-words, 3-gram, 5-gram, 7-gram"
echo "  # of Clusters: 20, 40, 60, 65, 80"
echo "  Random Seed: odd numbers in the range of [0, 40]"

N_CLUSTERS[0]=20
N_CLUSTERS[1]=40
N_CLUSTERS[2]=60
N_CLUSTERS[3]=65
N_CLUSTERS[4]=80

LOAD_FOLDER="tokens_extracted"
DATA_FILE[0]="bag"
DATA_FILE[1]="char3"
DATA_FILE[2]="char5"
DATA_FILE[3]="char7"
DATA_FILE_EXT=".csv"
CLASS_FILE="reuters21578.class"

OUTPUT_FOLDER="log"
OUTPUT_FILE_PREFIX="result_"
LOG_FILE_PREFIX="log_"

run_test() {
make kmeans
python ./preprocess.py
    for k in "${N_CLUSTERS[@]}"
    do
        for d in "${DATA_FILE[@]}"
        do
            inputfile="${LOAD_FOLDER}/${d}${DATA_FILE_EXT}"
            classfile="${LOAD_FOLDER}/${CLASS_FILE}"
            outfile_suf="$(date +"%H-%M-%S_%m-%d-%Y")"
            logfile="${OUTPUT_FOLDER}/${LOG_FILE_PREFIX}${k}_${d}_${outfile_suf}"
            outfile="${OUTPUT_FOLDER}/${OUTPUT_FILE_PREFIX}${k}_${d}_${outfile_suf}"
            # echo "Parameters:"
            # echo "  Data File:     ${d}"
            # echo "  # of Clusters: ${k}"
            # echo "  Output File:   ${OUTPUT_FILE_PREFIX}${k}_${d}_${outfile_suf}"
            # echo "  Log File:      ${LOG_FILE_PREFIX}${k}_${d}_${outfile_suf}"
./sphkmeans ${inputfile} ${classfile} ${k} 20 ${outfile} 2>&1 | tee ${logfile}
        done
    done
}

while
    echo "\nInput [Y]/[N] to start or exit test: "
    read inp
do
    if [ "$inp" == "Y" ] || [ "$inp" == "y" ]; then
        break
    elif [ "$inp" == "N" ] || [ "$inp" == "n" ]; then
        exit 1
    fi
done

run_test
