package com.learnai.words.cli;

import com.learnai.words.nn.OnnxLanguageModel;
import com.learnai.words.nn.TextGenerator;
import com.learnai.words.tokenizer.BPETokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Scanner;

/**
 * An interactive CLI to prompt the language model and generate text using ONNX Runtime.
 */
public class PromptCLI {
    public static void main(String[] args) {
        System.out.println("--- Interactive LLM Prompt CLI (ONNX) ---");

        BPETokenizer tokenizer = new BPETokenizer();
        Path tokPath = Path.of("tokenizer.bin");
        if (!Files.exists(tokPath)) {
            System.err.println("tokenizer.bin not found! Please train the model / tokenizer first.");
            return;
        }
        try {
            tokenizer.load(tokPath.toString());
        } catch (IOException e) {
            System.err.println("Failed to load tokenizer: " + e.getMessage());
            return;
        }

        Path modelPath = Path.of("model.onnx");
        if (!Files.exists(modelPath)) {
            System.err.println("model.onnx not found! Please train the model using train_model.sh first.");
            return;
        }

        int blockSize = Integer.getInteger("block.size", 256);

        System.out.println("Loading ONNX language model (" + modelPath.toAbsolutePath() + ")...");
        try (OnnxLanguageModel model = new OnnxLanguageModel(modelPath.toString())) {
            TextGenerator generator = new TextGenerator(model, tokenizer, blockSize);
            Scanner scanner = new Scanner(System.in);

            while (true) {
                System.out.print("\nEnter your prompt (or 'exit' to quit): ");
                String prompt = scanner.nextLine();
                if ("exit".equalsIgnoreCase(prompt.trim())) {
                    break;
                }

                if (prompt.isEmpty()) {
                    System.out.println("Prompt cannot be empty!");
                    continue;
                }

                System.out.print("Number of tokens to generate (default 50): ");
                String tokensInput = scanner.nextLine();
                int numTokens = 50;
                if (!tokensInput.trim().isEmpty()) {
                    try {
                        numTokens = Integer.parseInt(tokensInput.trim());
                    } catch (NumberFormatException e) {
                        System.out.println("Invalid number. Using default of 50.");
                    }
                }

                System.out.print("Temperature (default 0.7, lower = more deterministic): ");
                String tempInput = scanner.nextLine();
                double temp = 0.7;
                if (!tempInput.trim().isEmpty()) {
                    try {
                        temp = Double.parseDouble(tempInput.trim());
                    } catch (NumberFormatException e) {
                        System.out.println("Invalid temperature. Using default of 0.7.");
                    }
                }

                System.out.println("\nGenerating...");
                String result = generator.generate(prompt, numTokens, temp, 5);
                System.out.println("----------------------------------------");
                System.out.println(result);
                System.out.println("----------------------------------------");
            }
        } catch (Exception e) {
            System.err.println("Error initializing ONNX runtime session: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
