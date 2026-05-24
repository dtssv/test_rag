package com.rag.service.retrieval;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import com.rag.config.RagConfig;
import com.rag.service.llm.LlmClient;
import com.rag.service.rerank.RerankService;
import com.rag.service.rerank.RerankService.RerankCandidate;
import com.rag.service.rerank.RerankService.RerankedDocument;

/**
 * 混合召回融合策略服务
 * 将向量召回(Milvus)和BM25召回(Elasticsearch)结果进行融合
 * 支持加权融合(Weighted Fusion)和倒排融合(Reciprocal Rank Fusion, RRF)
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class HybridRetrievalService {

    private final MilvusVectorService milvusVectorService;
    private final ElasticsearchBm25Service elasticsearchBm25Service;
    private final LlmClient llmClient;
    private final RerankService rerankService;
    private final RagConfig ragConfig;

    /**
     * 混合召回：融合向量检索和BM25检索结果
     *
     * @param expandedQueries 扩写后的查询列表
     * @param topK            每个查询返回的文档数量
     * @param enableRerank    是否启用Rerank
     * @return 最终融合后的文档列表
     */
    public List<HybridResult> hybridRetrieve(List<String> expandedQueries, int topK, boolean enableRerank)
            throws IOException {
        RagConfig.HybridRetrievalConfig config = ragConfig.getHybridRetrieval();
        Map<String, HybridResult> mergedResults = new LinkedHashMap<>();

        // 对每个扩写查询分别进行向量检索和BM25检索
        for (String query : expandedQueries) {
            // 1. 向量检索
            float[] queryEmbedding = llmClient.getEmbedding(query);
            List<MilvusVectorService.VectorSearchResult> vectorResults = milvusVectorService.search(queryEmbedding,
                    topK,
                    com.rag.model.DocumentChunk.ChunkType.CHILD);

            // 2. BM25检索
            List<ElasticsearchBm25Service.Bm25SearchResult> bm25Results = elasticsearchBm25Service.search(query, topK,
                    "CHILD");

            // 3. 归一化分数并加权融合
            normalizeAndMerge(vectorResults, bm25Results, config, mergedResults);
        }

        // 4. 转为候选列表
        List<RerankCandidate> candidates = mergedResults.values().stream()
                .map(r -> RerankCandidate.builder()
                        .chunkId(r.getChunkId())
                        .parentChunkId(r.getParentChunkId())
                        .content(r.getContent())
                        .score(r.getFusedScore())
                        .source(r.getSource())
                        .build())
                .sorted(Comparator.comparingDouble(RerankCandidate::getScore).reversed())
                .collect(Collectors.toList());

        // 5. Rerank（如果启用）
        List<RerankedDocument> rerankedDocs = rerankService.rerank(
                expandedQueries.getFirst(), candidates, ragConfig.getRerank().getTopN());

        // 6. 获取父块内容
        List<String> parentIds = rerankedDocs.stream()
                .map(RerankedDocument::getParentChunkId)
                .filter(id -> id != null && !id.isEmpty())
                .distinct()
                .collect(Collectors.toList());

        Map<String, String> milvusParentMap = milvusVectorService.batchGetParentContent(parentIds);
        Map<String, String> esParentMap = elasticsearchBm25Service.batchGetParentContent(parentIds);
        Map<String, String> parentContentMap = new HashMap<>(milvusParentMap);
        parentContentMap.putAll(esParentMap);

        // 7. 构建最终结果（若为子块召回，补充父块内容）
        List<HybridResult> finalResults = new ArrayList<>();
        for (RerankedDocument doc : rerankedDocs) {
            String parentContent = null;
            if (doc.getParentChunkId() != null && !doc.getParentChunkId().isEmpty()) {
                parentContent = parentContentMap.get(doc.getParentChunkId());
            }

            finalResults.add(HybridResult.builder()
                    .chunkId(doc.getChunkId())
                    .parentChunkId(doc.getParentChunkId())
                    .content(doc.getContent())
                    .parentContent(parentContent)
                    .fusedScore(doc.getRerankScore())
                    .rerankScore(doc.getRerankScore())
                    .source(doc.getSource())
                    .build());
        }

        log.info("混合召回完成: {}个查询 -> {}个最终结果", expandedQueries.size(), finalResults.size());
        return finalResults;
    }

    /**
     * 归一化分数并加权融合向量结果和BM25结果
     */
    private void normalizeAndMerge(
            List<MilvusVectorService.VectorSearchResult> vectorResults,
            List<ElasticsearchBm25Service.Bm25SearchResult> bm25Results,
            RagConfig.HybridRetrievalConfig config,
            Map<String, HybridResult> mergedResults) {

        // 归一化向量分数（cosine similarity通常在0-1之间）
        double maxVectorScore = vectorResults.stream()
                .mapToDouble(MilvusVectorService.VectorSearchResult::getScore)
                .max().orElse(1.0);
        double minVectorScore = vectorResults.stream()
                .mapToDouble(MilvusVectorService.VectorSearchResult::getScore)
                .min().orElse(0.0);
        double vectorRange = maxVectorScore - minVectorScore;

        for (MilvusVectorService.VectorSearchResult r : vectorResults) {
            double normalizedScore = vectorRange > 0 ? (r.getScore() - minVectorScore) / vectorRange : r.getScore();
            double fusedScore = normalizedScore * config.getVectorWeight();

            mergedResults.merge(r.getChunkId(),
                    HybridResult.builder()
                            .chunkId(r.getChunkId())
                            .parentChunkId(r.getParentChunkId())
                            .content(r.getContent())
                            .fusedScore(fusedScore)
                            .source("VECTOR")
                            .build(),
                    (existing, newResult) -> {
                        existing.setFusedScore(existing.getFusedScore() + fusedScore);
                        existing.setSource("HYBRID");
                        return existing;
                    });
        }

        // 归一化BM25分数
        double maxBm25Score = bm25Results.stream()
                .mapToDouble(ElasticsearchBm25Service.Bm25SearchResult::getScore)
                .max().orElse(1.0);
        double minBm25Score = bm25Results.stream()
                .mapToDouble(ElasticsearchBm25Service.Bm25SearchResult::getScore)
                .min().orElse(0.0);
        double bm25Range = maxBm25Score - minBm25Score;

        for (ElasticsearchBm25Service.Bm25SearchResult r : bm25Results) {
            double normalizedScore = bm25Range > 0 ? (r.getScore() - minBm25Score) / bm25Range : r.getScore();
            double fusedScore = normalizedScore * config.getBm25Weight();

            mergedResults.merge(r.getChunkId(),
                    HybridResult.builder()
                            .chunkId(r.getChunkId())
                            .parentChunkId(r.getParentChunkId())
                            .content(r.getContent())
                            .fusedScore(fusedScore)
                            .source("BM25")
                            .build(),
                    (existing, newResult) -> {
                        existing.setFusedScore(existing.getFusedScore() + fusedScore);
                        existing.setSource("HYBRID");
                        return existing;
                    });
        }
    }

    /**
     * 混合召回结果
     */
    @lombok.Data
    @lombok.Builder
    @lombok.NoArgsConstructor
    @lombok.AllArgsConstructor
    public static class HybridResult {
        private String chunkId;
        private String parentChunkId;
        private String content;
        private String parentContent;
        private double fusedScore;
        private double rerankScore;
        private String source;
    }
}