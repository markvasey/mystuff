package com.markvasey.mysearch;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableAsync;

@SpringBootApplication
@EnableAsync
public class MySearchApplication {

    public static void main(String[] args) {
        SpringApplication.run(MySearchApplication.class, args);
    }
}
