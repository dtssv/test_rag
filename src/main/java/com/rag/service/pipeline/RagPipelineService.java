package com.rag.service.pipeline;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import main.java.com.rag.model.RagRequest;
import main.java.com.rag.model.RagResponse;
import main.java.com.rag.service.evaluation.EvaluationService;
import main.java.com.rag.service.evaluation.EvaluationService.EvaluationResult;
import main.java.com.rag.service.expansion.QueryExpansionService;
import main.java.com.rag.service.llm.LlmClient;
import main.java.com.rag.service.memory.MemoryService;
import main.java.com.rag.service.retrieval.HybridRetrievalService;
import main.java.com.rag.service.retrieval.HybridRetrievalService.HybridResult;

/**
 * RAG完整Pipeline服务
 * 串联所有模块，执行完整的RAG流程：
 *
 * 1. 记忆增强：结合短期/长期记忆增强查询
 * 2. 查询扩写：对查询生成多个扩展查询
 * 3. 混合召回：向量+BM25混合检索
 * 4. Rerank：对召回结果重排序
 * 5. 父块补充：通过子块关联父块获取更完整上下文
 * 6. 答案生成：基于召回上下文生成回答
 * 7. 记忆保存：保存短期/长期记忆
 * 8. 评测计算：计算各阶段评测指标
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RagPipelineService {

    private final QueryExpansionService queryExpansionService;
    private final HybridRetrievalService hybridRetrievalService;
    private final MemoryService memoryService;
    private final LlmClient llmClient;
    private final EvaluationService evaluationService;

    private static final String GENERATE_SYSTEM_PROMPT = """
            你是一个专业的知识问答助手。请根据以下参考资料回答用户的问题。
            回答要求：
            1. 只基于参考资料中的信息回答，不要编造不在资料中的内容
            2. 如果参考资料不足以回答问题，请如实说明
            3. 回答要准确、完整、有条理
            4. 使用中文回答
            """;

    /**
     * 执行完整的RAG Pipeline
     */
    public RagResponse execute(RagRequest request) {
        long startTime = System.currentTimeMillis();
        String sessionId = request.getSessionId();
        String query = request.getQuery();

        log.info("RAG Pipeline开始: sessionId={}, query={}", sessionId, query);

        // ===== Step 1: 记忆增强查询 =====
        String enhancedQuery = memoryService.buildMemoryEnhancedQuery(sessionId, query);

        // ===== Step 2: 获取长期记忆 =====
        List<String> longTermMemories = memoryService.retrieveLongTermMemory(enhancedQuery);
        List<String> shortTermMemories = memoryService.getShortTermMemoryTexts(sessionId);

        // ===== Step 3: 查询扩写 =====
        List<String> expandedQueries = queryExpansionService.expandQuery(enhancedQuery);
        log.info("查询扩写: {} -> {}个查询", query, expandedQueries.size());

        // ===== Step 4: 混合召回 =====
        long retrievalStartTime = System.currentTimeMillis();
        List<HybridResult> hybridResults;
        try {
            hybridResults = hybridRetrievalService.hybridRetrieve(
                    expandedQueries, request.getTopK() * 2, request.isEnableRerank());
        } catch (Exception e) {
            log.error("混合召回失败: {}", e.getMessage(), e);
            return buildErrorResponse(sessionId, query, expandedQueries, e.getMessage());
        }
        long retrievalLatencyMs = System.currentTimeMillis() - retrievalStartTime;

        // ===== Step 5: 构建上下文 =====
        // 优先使用父块内容（更大更完整），如果父块不存在则用子块内容
        String context = hybridResults.stream()
                .map(r -> r.getParentContent() != null ? r.getParentContent() : r.getContent())
                .distinct()
                .collect(Collectors.joining("\n\n"));

        // 加入长期记忆
        if (!longTermMemories.isEmpty()) {
            String memoryContext = "历史记忆:\n" + String.join("\n", longTermMemories);
            context = memoryContext + "\n\n参考资料:\n" + context;
        }

        // ===== Step 6: 答案生成 =====
        String userPrompt = "参考资料：\n" + context + "\n\n用户问题：" + query;
        String answer = llmClient.chat(GENERATE_SYSTEM_PROMPT, userPrompt);

        // ===== Step 7: 保存记忆 =====
        memoryService.saveShortTermMemory(sessionId, "user", query);
        memoryService.saveShortTermMemory(sessionId, "assistant", answer);
        memoryService.saveLongTermMemory(sessionId, query, answer);

        // ===== Step 8: 评测 =====
        long totalLatencyMs = System.currentTimeMillis() - startTime;
        List<String> retrievedContents = hybridResults.stream()
                .map(HybridResult::getContent)
                .collect(Collectors.toList());

        EvaluationResult evalResult = evaluationService.evaluate(
                query, answer, retrievedContents, null,
                retrievalLatencyMs, 0, totalLatencyMs);

        // ===== 构建响应 =====
        List<RagResponse.RetrievedDocument> retrievedDocs = hybridResults.stream()
                .map(r -> RagResponse.RetrievedDocument.builder()
                        .chunkId(r.getChunkId())
                        .parentContent(r.getParentContent())
                        .content(r.getContent())
                        .score(r.getFusedScore())
                        .rerankScore(r.getRerankScore())
                        .source(r.getSource())
                        .build())
                .collect(Collectors.toList());

        return RagResponse.builder()
                .sessionId(sessionId)
                .originalQuery(query)
                .expandedQueries(expandedQueries)
                .answer(answer)
                .retrievedDocuments(retrievedDocs)
                .usedMemory(RagResponse.MemoryContext.builder()
                        .shortTermMemories(shortTermMemories)
                        .longTermMemories(longTermMemories)
                        .build())
                .metrics(RagResponse.EvaluationMetrics.builder()
                        .retrievalPrecision(evalResult.getPrecision())
                        .retrievalRecall(evalResult.getRecall())
                        .retrievalF1(evalResult.getF1())
                        .contextRelevance(evalResult.getContextRelevance())
                        .answerFaithfulness(evalResult.getFaithfulness())
                        .answerRelevance(evalResult.getAnswerRelevance())
                        .retrievalLatencyMs(evalResult.getRetrievalLatencyMs())
                        .rerankLatencyMs(evalResult.getRerankLatencyMs())
                        .totalLatencyMs(evalResult.getTotalLatencyMs())
                        .build())
                .build();
    }

    private RagResponse buildErrorResponse(String sessionId, String query,
            List<String> expandedQueries, String errorMsg) {
        return RagResponse.builder()
                .sessionId(sessionId)
                .originalQuery(query)
                .expandedQueries(expandedQueries)
                .answer("系统处理异常：" + errorMsg)
                .retrievedDocuments(Collections.emptyList())
                .usedMemory(RagResponse.MemoryContext.builder()
                        .shortTermMemories(Collections.emptyList())
                        .longTermMemories(Collections.emptyList())
                        .build())
                .metrics(RagResponse.EvaluationMetrics.builder().build())
                .build();
    }
}