#!/bin/bash
export JAVA_HOME=/home/markvasey/.sdkman/candidates/java/26.0.1-tem
export PATH=$JAVA_HOME/bin:$PATH

./mvnw exec:exec -Dexec.args="--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Dd.model=256 -Dblock.size=128 -classpath %classpath com.learnai.words.cli.PromptCLI"
