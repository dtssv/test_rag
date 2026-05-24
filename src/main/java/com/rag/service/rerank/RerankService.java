package com.rag.service.rerank;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import com.rag.config.RagConfig;
import com.rag.service.llm.LlmClient;
import com.rag.service.llm.LlmClient.RerankResult;

/**
 * Rerank重排序服务
 * 对混合召回的结果进行精排序，提升最终结果质量
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RerankService {

    private final LlmClient llmClient;
    private final RagConfig ragConfig;

    /**
     * 对候选文档进行Rerank
     *
     * @param query      查询文本
     * @param candidates 候选文档列表
     * @param topN       返回前N个结果
     * @return 重排序后的文档列表
     */
    public List<RerankedDocument> rerank(String query, List<RerankCandidate> candidates, int topN) {
        RagConfig.RerankConfig config = ragConfig.getRerank();

        if (!config.isEnabled() || candidates == null || candidates.isEmpty()) {
            // Rerank未启用或无候选，直接返回
            return candidates.stream()
                    .map(c -> RerankedDocument.builder()
                            .chunkId(c.getChunkId())
                            .parentChunkId(c.getParentChunkId())
                            .content(c.getContent())
                            .originalScore(c.getScore())
                            .rerankScore(c.getScore())
                            .source(c.getSource())
                            .build())
                    .sorted(Comparator.comparingDouble(RerankedDocument::getRerankScore).reversed())
                    .limit(topN)
                    .collect(Collectors.toList());
        }

        try {
            // 提取文档内容用于rerank
            List<String> passages = candidates.stream()
                    .map(RerankCandidate::getContent)
                    .collect(Collectors.toList());

            // 调用Rerank模型
            List<RerankResult> rerankResults = llmClient.rerank(query, passages, topN);

            // 构建rerank后的结果
            List<RerankedDocument> rerankedDocs = new ArrayList<>();
            for (RerankResult rr : rerankResults) {
                if (rr.getIndex() < candidates.size()) {
                    RerankCandidate candidate = candidates.get(rr.getIndex());
                    rerankedDocs.add(RerankedDocument.builder()
                            .chunkId(candidate.getChunkId())
                            .parentChunkId(candidate.getParentChunkId())
                            .content(candidate.getContent())
                            .originalScore(candidate.getScore())
                            .rerankScore(rr.getRelevanceScore())
                            .source(candidate.getSource())
                            .build());
                }
            }

            log.debug("Rerank完成: {}个候选 -> {}个结果", candidates.size(), rerankedDocs.size());
            return rerankedDocs;

        } catch (Exception e) {
            log.warn("Rerank失败，使用原始排序: {}", e.getMessage());
            return candidates.stream()
                    .map(c -> RerankedDocument.builder()
                            .chunkId(c.getChunkId())
                            .parentChunkId(c.getParentChunkId())
                            .content(c.getContent())
                            .originalScore(c.getScore())
                            .rerankScore(c.getScore())
                            .source(c.getSource())
                            .build())
                    .sorted(Comparator.comparingDouble(RerankedDocument::getRerankScore).reversed())
                    .limit(topN)
                    .collect(Collectors.toList());
        }
    }

    /**
     * Rerank候选文档
     */
    @lombok.Data
    @lombok.Builder
    @lombok.NoArgsConstructor
    @lombok.AllArgsConstructor
    public static class RerankCandidate {
        private String chunkId;
        private String parentChunkId;
        private String content;
        private double score;
        private String source;
    }

    /**
     * Rerank后的文档
     */
    @lombok.Data
    @lombok.Builder
    @lombok.NoArgsConstructor
    @lombok.AllArgsConstructor
    public static class RerankedDocument {
        private String chunkId;
        private String parentChunkId;
        private String content;
        private double originalScore;
        private double rerankScore;
        private String source;
    }
}