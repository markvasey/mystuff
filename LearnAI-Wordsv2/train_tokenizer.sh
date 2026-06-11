#!/bin/bash
export JAVA_HOME=/home/markvasey/.sdkman/candidates/java/26.0.1-tem
export PATH=$JAVA_HOME/bin:$PATH

systemd-inhibit --what=sleep --why="BPE Tokenizer Training" ./mvnw exec:exec \
  -Dexec.args="--add-modules jdk.incubator.vector -Dtraining.dir=Training/TinyStories -Dvocab.size=4096 -classpath %classpath com.learnai.words.tokenizer.BPETrainTool"
