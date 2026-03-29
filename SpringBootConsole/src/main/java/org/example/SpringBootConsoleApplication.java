package org.example;

import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class SpringBootConsoleApplication implements CommandLineRunner {

    public static void main(String[] args) {
        SpringApplication.run(SpringBootConsoleApplication.class, args);
    }

    @Override
    public void run(String... args) throws Exception {
        System.out.println("Hello from the Spring Boot Console Application!");
        
        if (args.length > 0) {
            System.out.println("Arguments provided:");
            for (String arg : args) {
                System.out.println(" - " + arg);
            }
        } else {
            System.out.println("No arguments were provided.");
        }
    }
}
