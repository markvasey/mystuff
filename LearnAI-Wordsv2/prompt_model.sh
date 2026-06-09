#!/bin/bash
export JAVA_HOME=/home/markvasey/.sdkman/candidates/java/26.0.1-tem
export PATH=$JAVA_HOME/bin:$PATH

USE_GPU=${USE_GPU:-false}

./mvnw exec:exec -Dexec.args="--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Duse.gpu=$USE_GPU -Dd.model=128 -Dblock.size=64 -classpath %classpath com.learnai.words.cli.PromptCLI"
