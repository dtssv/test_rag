package com.rag.service.expansion;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import main.java.com.rag.config.RagConfig;
import main.java.com.rag.service.llm.LlmClient;

/**
 * 查询扩写服务
 * 对用户输入进行适当扩写，生成多个相关查询
 * 譬如用户输入"Redis问题怎么办"，扩写为：
 * - "Redis常见问题及解决方案"
 * - "Redis连接超时如何排查"
 * - "Redis内存溢出问题处理"
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class QueryExpansionService {

    private final LlmClient llmClient;
    private final RagConfig ragConfig;

    private static final String EXPANSION_SYSTEM_PROMPT = """
            你是一个查询扩写助手。用户会给你一个问题，你需要生成%d个相关的扩展查询。
            这些扩展查询应该：
            1. 覆盖原问题的不同方面和常见场景
            2. 包含更具体的问题描述，帮助检索更精准的信息
            3. 包含相关的技术术语和关键词
            4. 每个扩展查询独立成行，不要编号，不要额外解释

            直接输出扩展查询，每行一个，不要有任何其他内容。
            """;

    /**
     * 扩写查询
     *
     * @param originalQuery 原始查询
     * @return 扩写后的查询列表（包含原始查询）
     */
    public List<String> expandQuery(String originalQuery) {
        RagConfig.QueryExpansionConfig config = ragConfig.getQueryExpansion();

        if (!config.isEnabled()) {
            log.debug("查询扩写已禁用，返回原始查询");
            return List.of(originalQuery);
        }

        try {
            String systemPrompt = String.format(EXPANSION_SYSTEM_PROMPT, config.getExpansionCount());
            String userMessage = String.format("原始问题：%s", originalQuery);

            String response = llmClient.chat(systemPrompt, userMessage);

            List<String> expandedQueries = Arrays.stream(response.split("\n"))
                    .map(String::trim)
                    .filter(s -> !s.isEmpty())
                    .limit(config.getExpansionCount())
                    .collect(Collectors.toList());

            // 将原始查询放在首位
            List<String> allQueries = new ArrayList<>();
            allQueries.add(originalQuery);
            allQueries.addAll(expandedQueries);

            log.info("查询扩写完成: 原始查询[{}] -> {}个扩展查询", originalQuery, expandedQueries.size());
            return allQueries;

        } catch (Exception e) {
            log.warn("查询扩写失败，使用原始查询: {}", e.getMessage());
            return List.of(originalQuery);
        }
    }
}