package com.example.housepricemonitor;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableScheduling
public class HousePriceMonitorApplication {

	public static void main(String[] args) {
		SpringApplication.run(HousePriceMonitorApplication.class, args);
	}

}
