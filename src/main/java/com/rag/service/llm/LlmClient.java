package com.rag.service.llm;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.springframework.stereotype.Service;

import com.google.gson.*;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import com.rag.config.LlmConfig;
import okhttp3.*;

/**
 * LLM客户端，支持Embedding/Chat/Rerank调用
 * 兼容OpenAI API格式（Ollama/vLLM等均兼容）
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class LlmClient {

    private final LlmConfig llmConfig;
    private final Gson gson = new Gson();
    private final OkHttpClient httpClient = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build();

    /**
     * 调用Embedding模型，获取文本向量
     *
     * @param text 输入文本
     * @return 向量数组
     */
    public float[] getEmbedding(String text) {
        LlmConfig.EmbeddingConfig config = llmConfig.getEmbedding();
        String url = config.getBaseUrl() + "/v1/embeddings";

        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("model", config.getModel());
        requestBody.put("input", text);

        try {
            String response = doPost(url, requestBody);
            JsonObject json = JsonParser.parseString(response).getAsJsonObject();
            JsonArray embeddingArray = json.getAsJsonArray("data")
                    .get(0).getAsJsonObject()
                    .getAsJsonArray("embedding");

            float[] embedding = new float[embeddingArray.size()];
            for (int i = 0; i < embeddingArray.size(); i++) {
                embedding[i] = embeddingArray.get(i).getAsFloat();
            }
            return embedding;
        } catch (Exception e) {
            log.error("Embedding调用失败: {}", e.getMessage(), e);
            throw new RuntimeException("Embedding调用失败", e);
        }
    }

    /**
     * 批量获取Embedding
     *
     * @param texts 输入文本列表
     * @return 向量列表
     */
    public List<float[]> batchGetEmbedding(List<String> texts) {
        LlmConfig.EmbeddingConfig config = llmConfig.getEmbedding();
        String url = config.getBaseUrl() + "/v1/embeddings";

        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("model", config.getModel());
        requestBody.put("input", texts);

        try {
            String response = doPost(url, requestBody);
            JsonObject json = JsonParser.parseString(response).getAsJsonObject();
            JsonArray dataArray = json.getAsJsonArray("data");

            List<float[]> embeddings = new ArrayList<>();
            for (JsonElement element : dataArray) {
                JsonArray embeddingArray = element.getAsJsonObject().getAsJsonArray("embedding");
                float[] embedding = new float[embeddingArray.size()];
                for (int i = 0; i < embeddingArray.size(); i++) {
                    embedding[i] = embeddingArray.get(i).getAsFloat();
                }
                embeddings.add(embedding);
            }
            return embeddings;
        } catch (Exception e) {
            log.error("批量Embedding调用失败: {}", e.getMessage(), e);
            throw new RuntimeException("批量Embedding调用失败", e);
        }
    }

    /**
     * 调用Chat模型
     *
     * @param systemPrompt 系统提示词
     * @param userMessage  用户消息
     * @return 模型回复文本
     */
    public String chat(String systemPrompt, String userMessage) {
        LlmConfig.ChatConfig config = llmConfig.getChat();
        String url = config.getBaseUrl() + "/v1/chat/completions";

        List<Map<String, String>> messages = new ArrayList<>();
        if (systemPrompt != null && !systemPrompt.isEmpty()) {
            messages.add(Map.of("role", "system", "content", systemPrompt));
        }
        messages.add(Map.of("role", "user", "content", userMessage));

        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("model", config.getModel());
        requestBody.put("messages", messages);
        requestBody.put("max_tokens", config.getMaxTokens());
        requestBody.put("temperature", config.getTemperature());

        try {
            String response = doPost(url, requestBody);
            JsonObject json = JsonParser.parseString(response).getAsJsonObject();
            return json.getAsJsonArray("choices")
                    .get(0).getAsJsonObject()
                    .getAsJsonObject("message")
                    .get("content").getAsString();
        } catch (Exception e) {
            log.error("Chat调用失败: {}", e.getMessage(), e);
            throw new RuntimeException("Chat调用失败", e);
        }
    }

    /**
     * 调用Rerank模型
     *
     * @param query    查询文本
     * @param passages 待排序的文档列表
     * @param topN     返回前N个结果
     * @return 排序后的结果列表，每个元素包含index和relevance_score
     */
    public List<RerankResult> rerank(String query, List<String> passages, int topN) {
        LlmConfig.RerankConfig config = llmConfig.getRerank();
        String url = config.getBaseUrl() + "/v1/rerank";

        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("model", config.getModel());
        requestBody.put("query", query);
        requestBody.put("documents", passages);
        requestBody.put("top_n", Math.min(topN, passages.size()));

        try {
            String response = doPost(url, requestBody);
            JsonObject json = JsonParser.parseString(response).getAsJsonObject();
            JsonArray results = json.getAsJsonArray("results");

            List<RerankResult> rerankResults = new ArrayList<>();
            for (JsonElement element : results) {
                JsonObject obj = element.getAsJsonObject();
                rerankResults.add(RerankResult.builder()
                        .index(obj.get("index").getAsInt())
                        .relevanceScore(obj.get("relevance_score").getAsDouble())
                        .build());
            }
            return rerankResults;
        } catch (Exception e) {
            log.error("Rerank调用失败: {}", e.getMessage(), e);
            throw new RuntimeException("Rerank调用失败", e);
        }
    }

    private String doPost(String url, Map<String, Object> requestBody) throws IOException {
        String jsonBody = gson.toJson(requestBody);
        Request request = new Request.Builder()
                .url(url)
                .post(RequestBody.create(jsonBody, MediaType.parse("application/json")))
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String errorBody = response.body() != null ? response.body().string() : "no body";
                throw new IOException("API调用失败, code=" + response.code() + ", body=" + errorBody);
            }
            return response.body() != null ? response.body().string() : "";
        }
    }

    /**
     * Rerank结果
     */
    @lombok.Data
    @lombok.Builder
    @lombok.NoArgsConstructor
    @lombok.AllArgsConstructor
    public static class RerankResult {
        private int index;
        private double relevanceScore;
    }
}