package com.rag.model;

import java.util.List;
import java.util.Map;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * RAG响应模型
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RagResponse {

    /**
     * 会话ID
     */
    private String sessionId;

    /**
     * 原始用户查询
     */
    private String originalQuery;

    /**
     * 扩写后的查询列表
     */
    private List<String> expandedQueries;

    /**
     * 最终生成的回答
     */
    private String answer;

    /**
     * 召回的文档块列表（经过rerank后的最终结果）
     */
    private List<RetrievedDocument> retrievedDocuments;

    /**
     * 使用的记忆信息
     */
    private MemoryContext usedMemory;

    /**
     * 评测指标
     */
    private EvaluationMetrics metrics;

    /**
     * 召回的文档结果
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class RetrievedDocument {
        /**
         * 文档块ID
         */
        private String chunkId;

        /**
         * 父块内容（若为子块召回，则填充对应父块内容）
         */
        private String parentContent;

        /**
         * 当前块内容
         */
        private String content;

        /**
         * 综合得分（融合后的分数）
         */
        private double score;

        /**
         * rerank得分
         */
        private Double rerankScore;

        /**
         * 召回来源: VECTOR / BM25 / HYBRID
         */
        private String source;

        /**
         * 元数据
         */
        private Map<String, Object> metadata;
    }

    /**
     * 使用的记忆上下文
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class MemoryContext {
        private List<String> shortTermMemories;
        private List<String> longTermMemories;
    }

    /**
     * 评测指标
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class EvaluationMetrics {
        private double retrievalPrecision;
        private double retrievalRecall;
        private double retrievalF1;
        private double contextRelevance;
        private double answerFaithfulness;
        private double answerRelevance;
        private long retrievalLatencyMs;
        private long rerankLatencyMs;
        private long totalLatencyMs;
    }
}