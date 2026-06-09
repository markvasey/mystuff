#!/bin/bash
export JAVA_HOME=/home/markvasey/.sdkman/candidates/java/26.0.1-tem
export PATH=$JAVA_HOME/bin:$PATH

USE_GPU=${USE_GPU:-false}

systemd-inhibit --what=sleep --why="Language Model Training" ./mvnw clean compile exec:exec \
  -Dexec.args="--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Duse.gpu=$USE_GPU -Dtraining.dir=Training/TinyStories -Dblock.size=64 -Dd.model=128 -Depochs=40 -Dlearning.rate=0.001 -classpath %classpath com.learnai.words.cli.WordsCLI"
