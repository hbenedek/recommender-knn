#!/usr/bin/env bash
# If your default java install does not work, explicitly 
# provide the path to the JDK 1.8 installation. On OSX
# with homebrew:
# export JAVA_HOME=/usr/local/Cellar/openjdk@8/1.8.0+282;
export JAVA_OPTS="-Xmx8G";
RUN=./logs/timeOthers-$(date "+%Y-%m-%d-%H:%M:%S")-$(hostname)
mkdir -p $RUN
LOGS=$RUN/log.txt
echo "------------------- BASELINE    ---------------------" >> $LOGS
sbt "runMain predict.Baseline --train data/ml-100k/u2.base --test data/ml-100k/u2.test --json $RUN/baseline-100k.json --num_measurements 3" 2>&1 >>$LOGS
echo "------------------- DISTRIBUTED ---------------------" >> $LOGS
sbt "runMain predict.Baseline --train data/ml-25m/r2.train --test data/ml-25m/r2.test --separator , --json $RUN/baseline-25m.json --num_measurements 3" 2>&1 >> $LOGS
sbt "runMain distributed.DistributedBaseline --train data/ml-25m/r2.train --test data/ml-25m/r2.test --separator , --json $RUN/distributed-25m-1.json --num_measurements 3 --master local[1]" 2>&1 >>$LOGS
sbt "runMain distributed.DistributedBaseline --train data/ml-25m/r2.train --test data/ml-25m/r2.test --separator , --json $RUN/distributed-25m-4.json --num_measurements 3 --master local[4]" 2>&1 >>$LOGS
