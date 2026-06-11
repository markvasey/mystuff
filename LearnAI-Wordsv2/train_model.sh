#!/bin/bash
export JAVA_HOME=/home/markvasey/.sdkman/candidates/java/26.0.1-tem
export PATH=$JAVA_HOME/bin:$PATH

systemd-inhibit --what=sleep --why="Language Model Training" ./mvnw clean compile exec:exec \
  -Dexec.args="--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Dtraining.dir=Training/TinyStories -Dblock.size=128 -Dbatch.size.train=512 -Dd.model=256 -Depochs=40 -Dlearning.rate=0.0003 -classpath %classpath com.learnai.words.cli.WordsCLI"
