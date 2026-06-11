#!/bin/bash
export JAVA_HOME=/home/markvasey/.sdkman/candidates/java/26.0.1-tem
export PATH=$JAVA_HOME/bin:$PATH

systemd-inhibit --what=sleep --why="Language Model Training" ./mvnw clean compile exec:exec \
  -Dexec.args="--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Dtraining.dir=Training/FictionalLiterature -Dblock.size=256 -Dbatch.size.train=512 -Dd.model=256 -Dlearning.rate=0.0001 -classpath %classpath com.learnai.words.cli.WordsCLI"
