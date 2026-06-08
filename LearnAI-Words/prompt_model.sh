#!/bin/bash
export JAVA_HOME=/home/markvasey/.sdkman/candidates/java/26.0.1-tem
export PATH=$JAVA_HOME/bin:$PATH

./mvnw exec:exec -Dexec.args="--add-modules jdk.incubator.vector -Dd.model=128 -Dblock.size=64 -classpath %classpath com.learnai.words.cli.PromptCLI"
