package com.rag.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import lombok.Data;

@Data
@Configuration
@ConfigurationProperties(prefix = "llm")
public class LlmConfig {

    private EmbeddingConfig embedding = new EmbeddingConfig();
    private ChatConfig chat = new ChatConfig();
    private RerankConfig rerank = new RerankConfig();

    @Data
    public static class EmbeddingConfig {
        private String baseUrl = "http://localhost:11434";
        private String model = "bge-m3";
        private int dimension = 1024;
    }

    @Data
    public static class ChatConfig {
        private String baseUrl = "http://localhost:11434";
        private String model = "qwen2.5:7b";
        private int maxTokens = 4096;
        private double temperature = 0.7;
    }

    @Data
    public static class RerankConfig {
        private String baseUrl = "http://localhost:11434";
        private String model = "bge-reranker-v2-m3";
        private int topN = 5;
    }
}