package com.rag.service.memory;

import java.lang.reflect.Type;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import main.java.com.rag.config.RagConfig;
import main.java.com.rag.service.llm.LlmClient;
import main.java.com.rag.service.retrieval.MilvusVectorService;

/**
 * 记忆服务
 * 包含短期记忆(Redis)和长期记忆(Milvus向量检索)
 *
 * 短期记忆：存储当前会话的最近对话历史，有TTL过期时间
 * 长期记忆：将重要的对话信息向量化存储，跨会话持久化
 *
 * 记忆增强召回：在向量召回时结合记忆信息，使得后续轮次中有意义但之前输入的信息
 * 仍然可以被召回使用
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class MemoryService {

    private final StringRedisTemplate redisTemplate;
    private final LlmClient llmClient;
    private final MilvusVectorService milvusVectorService;
    private final RagConfig ragConfig;
    private final Gson gson = new Gson();

    private static final String SHORT_TERM_KEY_PREFIX = "rag:memory:short:";
    private static final String LONG_TERM_COLLECTION = "rag_long_term_memory";
    private static final int EMBEDDING_DIMENSION = 1024;

    /**
     * 对话消息
     */
    public record ChatMessage(String role, String content, long timestamp) {
    }

    // ==================== 短期记忆 ====================

    /**
     * 保存短期记忆（当前会话的对话历史）
     *
     * @param sessionId 会话ID
     * @param role      角色(user/assistant)
     * @param content   消息内容
     */
    public void saveShortTermMemory(String sessionId, String role, String content) {
        RagConfig.MemoryConfig.ShortTermConfig config = ragConfig.getMemory().getShortTerm();
        if (!config.isEnabled())
            return;

        String key = SHORT_TERM_KEY_PREFIX + sessionId;
        ChatMessage message = new ChatMessage(role, content, System.currentTimeMillis());

        // 追加消息到列表
        redisTemplate.opsForList().rightPush(key, gson.toJson(message));

        // 保持最多maxTurns条消息（2条消息=1轮对话，所以限制2*maxTurns）
        long maxSize = config.getMaxTurns() * 2L;
        while (redisTemplate.opsForList().size(key) > maxSize) {
            redisTemplate.opsForList().leftPop(key);
        }

        // 设置过期时间
        redisTemplate.expire(key, config.getTtlMinutes(), TimeUnit.MINUTES);
        log.debug("保存短期记忆: sessionId={}, role={}", sessionId, role);
    }

    /**
     * 获取短期记忆（当前会话的对话历史）
     *
     * @param sessionId 会话ID
     * @return 对话历史列表
     */
    public List<ChatMessage> getShortTermMemory(String sessionId) {
        RagConfig.MemoryConfig.ShortTermConfig config = ragConfig.getMemory().getShortTerm();
        if (!config.isEnabled())
            return Collections.emptyList();

        String key = SHORT_TERM_KEY_PREFIX + sessionId;
        List<String> messages = redisTemplate.opsForList().range(key, 0, -1);

        if (messages == null || messages.isEmpty()) {
            return Collections.emptyList();
        }

        Type type = new TypeToken<ChatMessage>() {
        }.getType();
        return messages.stream()
                .map(json -> gson.fromJson(json, type))
                .collect(Collectors.toList());
    }

    /**
     * 获取短期记忆文本（用于构建上下文）
     */
    public List<String> getShortTermMemoryTexts(String sessionId) {
        List<ChatMessage> messages = getShortTermMemory(sessionId);
        return messages.stream()
                .map(m -> m.role() + ": " + m.content())
                .collect(Collectors.toList());
    }

    /**
     * 清除短期记忆
     */
    public void clearShortTermMemory(String sessionId) {
        String key = SHORT_TERM_KEY_PREFIX + sessionId;
        redisTemplate.delete(key);
        log.info("清除短期记忆: sessionId={}", sessionId);
    }

    // ==================== 长期记忆 ====================

    /**
     * 保存长期记忆
     * 将重要的对话信息提取后向量化存储到Milvus
     *
     * @param sessionId 会话ID
     * @param query     用户问题
     * @param answer    系统回答
     */
    public void saveLongTermMemory(String sessionId, String query, String answer) {
        RagConfig.MemoryConfig.LongTermConfig config = ragConfig.getMemory().getLongTerm();
        if (!config.isEnabled())
            return;

        try {
            // 使用LLM提取关键信息作为长期记忆
            String memoryContent = extractMemoryContent(query, answer);
            if (memoryContent == null || memoryContent.isEmpty())
                return;

            // 获取向量
            float[] embedding = llmClient.getEmbedding(memoryContent);

            // 构建记忆文档块
            com.rag.model.DocumentChunk memoryChunk = com.rag.model.DocumentChunk.builder()
                    .chunkId("memory_" + sessionId + "_" + System.currentTimeMillis())
                    .parentChunkId(null)
                    .documentId("memory_" + sessionId)
                    .chunkType(com.rag.model.DocumentChunk.ChunkType.PARENT)
                    .content(memoryContent)
                    .embedding(embedding)
                    .metadata(Map.of(
                            "type", "long_term_memory",
                            "sessionId", sessionId,
                            "originalQuery", query))
                    .build();

            // 存储到Milvus
            milvusVectorService.batchInsert(List.of(memoryChunk));
            log.info("保存长期记忆: sessionId={}, content={}", sessionId,
                    memoryContent.substring(0, Math.min(50, memoryContent.length())));

        } catch (Exception e) {
            log.warn("保存长期记忆失败: {}", e.getMessage());
        }
    }

    /**
     * 检索长期记忆
     * 通过向量相似度检索与当前查询相关的历史记忆
     *
     * @param query 当前查询
     * @return 相关的长期记忆列表
     */
    public List<String> retrieveLongTermMemory(String query) {
        RagConfig.MemoryConfig.LongTermConfig config = ragConfig.getMemory().getLongTerm();
        if (!config.isEnabled())
            return Collections.emptyList();

        try {
            float[] queryEmbedding = llmClient.getEmbedding(query);

            // 在Milvus中搜索相关记忆
            List<MilvusVectorService.VectorSearchResult> results = milvusVectorService.search(queryEmbedding,
                    config.getMaxMemories(),
                    com.rag.model.DocumentChunk.ChunkType.PARENT);

            // 过滤掉低于相似度阈值的结果
            return results.stream()
                    .filter(r -> r.getScore() >= config.getSimilarityThreshold())
                    .map(MilvusVectorService.VectorSearchResult::getContent)
                    .collect(Collectors.toList());

        } catch (Exception e) {
            log.warn("检索长期记忆失败: {}", e.getMessage());
            return Collections.emptyList();
        }
    }

    /**
     * 使用LLM从对话中提取值得长期记忆的关键信息
     */
    private String extractMemoryContent(String query, String answer) {
        String prompt = "请从以下对话中提取值得长期记住的关键信息（如用户偏好、重要事实、决策结论等）。" +
                "如果对话中没有值得长期记忆的信息，请回复空字符串。\n\n" +
                "用户问题：" + query + "\n" +
                "系统回答：" + answer + "\n\n" +
                "关键信息：";

        String result = llmClient.chat(null, prompt);
        return result != null ? result.trim() : "";
    }

    // ==================== 记忆增强 ====================

    /**
     * 构建记忆增强的查询
     * 将短期记忆和长期记忆融入查询上下文，避免后续轮次中有价值信息丢失
     *
     * @param sessionId    会话ID
     * @param currentQuery 当前查询
     * @return 增强后的查询上下文
     */
    public String buildMemoryEnhancedQuery(String sessionId, String currentQuery) {
        StringBuilder enhancedQuery = new StringBuilder(currentQuery);

        // 获取短期记忆中的最近用户输入
        List<String> shortTermTexts = getShortTermMemoryTexts(sessionId);
        if (!shortTermTexts.isEmpty()) {
            // 取最近几轮对话作为上下文
            int startIdx = Math.max(0, shortTermTexts.size() - 4);
            List<String> recentContext = shortTermTexts.subList(startIdx, shortTermTexts.size());
            String contextStr = String.join("\n", recentContext);

            // 使用LLM判断是否需要补充上下文
            String contextPrompt = "根据以下对话历史，判断当前问题是否需要历史上下文来理解。" +
                    "如果需要，请补充必要的上下文信息到当前问题中；如果不需要，直接返回当前问题。\n\n" +
                    "对话历史：\n" + contextStr + "\n\n" +
                    "当前问题：" + currentQuery + "\n\n" +
                    "增强后的问题：";

            try {
                String result = llmClient.chat(null, contextPrompt);
                if (result != null && !result.trim().isEmpty()) {
                    enhancedQuery = new StringBuilder(result.trim());
                    log.debug("记忆增强查询: [{}] -> [{}]", currentQuery, enhancedQuery);
                }
            } catch (Exception e) {
                log.warn("记忆增强查询失败，使用原始查询: {}", e.getMessage());
            }
        }

        return enhancedQuery.toString();
    }
}