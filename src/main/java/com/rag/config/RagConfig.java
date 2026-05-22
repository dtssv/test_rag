package com.rag.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import lombok.Data;

@Data
@Configuration
@ConfigurationProperties(prefix = "rag")
public class RagConfig {

    private ChunkConfig chunk = new ChunkConfig();
    private QueryExpansionConfig queryExpansion = new QueryExpansionConfig();
    private HybridRetrievalConfig hybridRetrieval = new HybridRetrievalConfig();
    private MemoryConfig memory = new MemoryConfig();
    private RerankConfig rerank = new RerankConfig();
    private EvaluationConfig evaluation = new EvaluationConfig();

    @Data
    public static class ChunkConfig {
        private int parentSize = 512;
        private int parentOverlap = 50;
        private int childSize = 128;
        private int childOverlap = 20;
    }

    @Data
    public static class QueryExpansionConfig {
        private boolean enabled = true;
        private int expansionCount = 3;
    }

    @Data
    public static class HybridRetrievalConfig {
        private double vectorWeight = 0.7;
        private double bm25Weight = 0.3;
    }

    @Data
    public static class MemoryConfig {
        private ShortTermConfig shortTerm = new ShortTermConfig();
        private LongTermConfig longTerm = new LongTermConfig();

        @Data
        public static class ShortTermConfig {
            private boolean enabled = true;
            private int maxTurns = 10;
            private int ttlMinutes = 60;
        }

        @Data
        public static class LongTermConfig {
            private boolean enabled = true;
            private double similarityThreshold = 0.7;
            private int maxMemories = 5;
        }
    }

    @Data
    public static class RerankConfig {
        private boolean enabled = true;
        private int topN = 5;
    }

    @Data
    public static class EvaluationConfig {
        private boolean enabled = true;
    }
}