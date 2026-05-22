package com.rag.service.evaluation;

import java.util.List;
import java.util.Set;

import org.springframework.stereotype.Service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import main.java.com.rag.config.RagConfig;
import main.java.com.rag.service.llm.LlmClient;

/**
 * RAG评测指标服务
 * 提供检索和生成阶段的核心评测指标：
 *
 * 检索阶段：
 * - Precision: 召回文档中相关文档的比例
 * - Recall: 相关文档中被召回的比例
 * - F1: Precision和Recall的调和平均
 * - Context Relevance: 召回上下文与查询的相关性
 *
 * 生成阶段：
 * - Faithfulness: 生成回答与召回上下文的一致性（是否产生幻觉）
 * - Answer Relevance: 生成回答与原始问题的相关性
 *
 * 性能指标：
 * - 检索延迟
 * - Rerank延迟
 * - 总延迟
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class EvaluationService {

    private final LlmClient llmClient;
    private final RagConfig ragConfig;

    /**
     * 计算检索阶段的评测指标
     *
     * @param query          查询文本
     * @param retrievedDocs  召回的文档列表
     * @param relevantDocIds 标注的相关文档ID集合（可选，用于有标注数据的评测）
     * @return 评测指标
     */
    public RetrievalMetrics evaluateRetrieval(String query, List<String> retrievedDocs,
            Set<String> relevantDocIds) {
        double precision = 0.0;
        double recall = 0.0;
        double f1 = 0.0;

        if (relevantDocIds != null && !relevantDocIds.isEmpty() && retrievedDocs != null) {
            // 有标注数据时，计算精确的P/R/F1
            long relevantRetrieved = retrievedDocs.stream()
                    .filter(relevantDocIds::contains)
                    .count();

            precision = retrievedDocs.isEmpty() ? 0 : (double) relevantRetrieved / retrievedDocs.size();
            recall = (double) relevantRetrieved / relevantDocIds.size();
            f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;
        }

        // 无标注数据时，使用LLM评估上下文相关性
        double contextRelevance = evaluateContextRelevance(query, retrievedDocs);

        return RetrievalMetrics.builder()
                .precision(precision)
                .recall(recall)
                .f1(f1)
                .contextRelevance(contextRelevance)
                .build();
    }

    /**
     * 使用LLM评估上下文相关性（无标注数据时使用）
     * 判断召回的文档是否与查询相关
     */
    private double evaluateContextRelevance(String query, List<String> retrievedDocs) {
        if (retrievedDocs == null || retrievedDocs.isEmpty())
            return 0.0;

        // 取前5个文档进行评估，避免prompt过长
        List<String> docsToEval = retrievedDocs.size() > 5 ? retrievedDocs.subList(0, 5) : retrievedDocs;

        String docsText = String.join("\n---\n", docsToEval);

        String prompt = "请评估以下召回文档与查询的相关性。返回一个0到1之间的分数，" +
                "1表示完全相关，0表示完全不相关。只返回数字，不要其他内容。\n\n" +
                "查询：" + query + "\n\n" +
                "召回文档：\n" + docsText + "\n\n" +
                "相关性分数：";

        try {
            String result = llmClient.chat(null, prompt);
            return Double.parseDouble(result.trim());
        } catch (Exception e) {
            log.warn("上下文相关性评估失败: {}", e.getMessage());
            return 0.5; // 默认中间值
        }
    }

    /**
     * 评估生成回答的Faithfulness（忠实度）
     * 判断回答是否基于召回上下文，是否产生幻觉
     *
     * @param answer   生成的回答
     * @param contexts 召回的上下文文档
     * @return 忠实度分数(0-1)
     */
    public double evaluateFaithfulness(String answer, List<String> contexts) {
        if (contexts == null || contexts.isEmpty())
            return 0.0;

        String contextText = String.join("\n---\n", contexts);

        String prompt = "请评估以下回答是否忠实于给定的上下文信息。" +
                "如果回答中的信息都能在上下文中找到依据，则忠实度高；" +
                "如果回答包含上下文中没有的信息（幻觉），则忠实度低。" +
                "返回一个0到1之间的分数，1表示完全忠实，0表示完全不可信。" +
                "只返回数字，不要其他内容。\n\n" +
                "上下文：\n" + contextText + "\n\n" +
                "回答：" + answer + "\n\n" +
                "忠实度分数：";

        try {
            String result = llmClient.chat(null, prompt);
            return Double.parseDouble(result.trim());
        } catch (Exception e) {
            log.warn("Faithfulness评估失败: {}", e.getMessage());
            return 0.5;
        }
    }

    /**
     * 评估生成回答的Relevance（相关性）
     * 判断回答是否与原始问题相关
     *
     * @param query  原始问题
     * @param answer 生成的回答
     * @return 相关性分数(0-1)
     */
    public double evaluateAnswerRelevance(String query, String answer) {
        String prompt = "请评估以下回答是否与问题相关。" +
                "返回一个0到1之间的分数，1表示完全相关，0表示完全不相关。" +
                "只返回数字，不要其他内容。\n\n" +
                "问题：" + query + "\n\n" +
                "回答：" + answer + "\n\n" +
                "相关性分数：";

        try {
            String result = llmClient.chat(null, prompt);
            return Double.parseDouble(result.trim());
        } catch (Exception e) {
            log.warn("Answer Relevance评估失败: {}", e.getMessage());
            return 0.5;
        }
    }

    /**
     * 综合评测
     *
     * @param query              查询
     * @param answer             回答
     * @param retrievedDocs      召回文档
     * @param relevantDocIds     标注的相关文档ID（可选）
     * @param retrievalLatencyMs 检索延迟
     * @param rerankLatencyMs    Rerank延迟
     * @param totalLatencyMs     总延迟
     * @return 完整的评测指标
     */
    public EvaluationResult evaluate(String query, String answer,
            List<String> retrievedDocs,
            Set<String> relevantDocIds,
            long retrievalLatencyMs,
            long rerankLatencyMs,
            long totalLatencyMs) {
        if (!ragConfig.getEvaluation().isEnabled()) {
            return EvaluationResult.builder()
                    .retrievalLatencyMs(retrievalLatencyMs)
                    .rerankLatencyMs(rerankLatencyMs)
                    .totalLatencyMs(totalLatencyMs)
                    .build();
        }

        // 检索阶段评测
        RetrievalMetrics retrievalMetrics = evaluateRetrieval(query, retrievedDocs, relevantDocIds);

        // 生成阶段评测
        double faithfulness = evaluateFaithfulness(answer, retrievedDocs);
        double answerRelevance = evaluateAnswerRelevance(query, answer);

        return EvaluationResult.builder()
                .precision(retrievalMetrics.getPrecision())
                .recall(retrievalMetrics.getRecall())
                .f1(retrievalMetrics.getF1())
                .contextRelevance(retrievalMetrics.getContextRelevance())
                .faithfulness(faithfulness)
                .answerRelevance(answerRelevance)
                .retrievalLatencyMs(retrievalLatencyMs)
                .rerankLatencyMs(rerankLatencyMs)
                .totalLatencyMs(totalLatencyMs)
                .build();
    }

    /**
     * 检索阶段指标
     */
    @lombok.Data
    @lombok.Builder
    @lombok.NoArgsConstructor
    @lombok.AllArgsConstructor
    public static class RetrievalMetrics {
        private double precision;
        private double recall;
        private double f1;
        private double contextRelevance;
    }

    /**
     * 完整评测结果
     */
    @lombok.Data
    @lombok.Builder
    @lombok.NoArgsConstructor
    @lombok.AllArgsConstructor
    public static class EvaluationResult {
        private double precision;
        private double recall;
        private double f1;
        private double contextRelevance;
        private double faithfulness;
        private double answerRelevance;
        private long retrievalLatencyMs;
        private long rerankLatencyMs;
        private long totalLatencyMs;
    }
}