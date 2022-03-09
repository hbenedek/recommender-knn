#!/usr/bin/env bash
# If your default java install does not work, explicitly 
# provide the path to the JDK 1.8 installation. On OSX
# with homebrew:
# export JAVA_HOME=/usr/local/Cellar/openjdk@8/1.8.0+282; ./run.sh
export JAVA_OPTS="-Xmx8G";
RUN=./logs/run-$(date "+%Y-%m-%d-%H:%M:%S")-$(hostname)
mkdir -p $RUN
LOGS=$RUN/log.txt
source ./config.sh 
echo "------------------- BASELINE    ---------------------" >> $LOGS
sbt "runMain predict.Baseline --train $ML100Ku2base --test $ML100Ku2test --json $RUN/baseline-100k.json" 2>&1 >>$LOGS
echo "------------------- DISTRIBUTED ---------------------" >> $LOGS
sbt "runMain predict.Baseline --train $ML25Mr2train --test $ML25Mr2test --separator , --json $RUN/baseline-25m.json" 2>&1 >>$LOGS
sbt "runMain distributed.DistributedBaseline --train $ML25Mr2train --test $ML25Mr2test --separator , --json $RUN/distributed-25m-4.json --master $SPARKMASTER" 2>&1 >>$LOGS
echo "------------------- PERSONALIZED --------------------" >> $LOGS
sbt "runMain predict.Personalized --train $ML100Ku2base --test $ML100Ku2test --json $RUN/personalized-100k.json" 2>&1 >>$LOGS
echo "------------------- KNN -----------------------------" >> $LOGS
sbt "runMain predict.kNN --train $ML100Ku2base --test $ML100Ku2test --json $RUN/knn-100k.json" 2>&1 >>$LOGS
echo "------------------- RECOMMEND -----------------------" >> $LOGS
sbt "runMain recommend.Recommender --data $ML100Kudata --personal data/personal.csv --json $RUN/recommender-100k.json" 2>&1 >>$LOGS
