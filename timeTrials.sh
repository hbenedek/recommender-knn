#!/usr/bin/env bash
# If your default java install does not work, explicitly 
# provide the path to the JDK 1.8 installation. On OSX
# with homebrew:
# export JAVA_HOME=/usr/local/Cellar/openjdk@8/1.8.0+282;
export JAVA_OPTS="-Xmx8G";
RUN=./logs/timetrials-$(date "+%Y-%m-%d-%H:%M:%S")-$(hostname)
mkdir -p $RUN
LOGS=$RUN/log.txt
echo "------------------- KNN -----------------------------" >> $LOGS
sbt "runMain predict.kNN --train data/ml-100k/u2.base --test data/ml-100k/u2.test --json $RUN/knn-100k.json --num_measurements 3" 2>&1 >>$LOGS
