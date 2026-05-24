package com.rag.controller;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import com.rag.model.DocumentChunk;
import com.rag.model.RagRequest;
import com.rag.model.RagResponse;
import com.rag.service.chunk.DocumentChunkService;
import com.rag.service.llm.LlmClient;
import com.rag.service.memory.MemoryService;
import com.rag.service.pipeline.RagPipelineService;
import com.rag.service.retrieval.ElasticsearchBm25Service;
import com.rag.service.retrieval.MilvusVectorService;

/**
 * RAG系统REST API接口
 */
@Slf4j
@RestController
@RequestMapping("/api/rag")
@RequiredArgsConstructor
public class RagController {

    private final RagPipelineService ragPipelineService;
    private final DocumentChunkService documentChunkService;
    private final MilvusVectorService milvusVectorService;
    private final ElasticsearchBm25Service elasticsearchBm25Service;
    private final LlmClient llmClient;
    private final MemoryService memoryService;

    /**
     * RAG问答接口
     * POST /api/rag/query
     */
    @PostMapping("/query")
    public ResponseEntity<RagResponse> query(@Valid @RequestBody RagRequest request) {
        log.info("收到RAG查询请求: sessionId={}, query={}", request.getSessionId(), request.getQuery());
        RagResponse response = ragPipelineService.execute(request);
        return ResponseEntity.ok(response);
    }

    /**
     * 文档入库接口
     * POST /api/rag/documents
     */
    @PostMapping("/documents")
    public ResponseEntity<Map<String, Object>> ingestDocument(
            @RequestBody DocumentIngestRequest request) throws IOException {

        String documentId = request.getDocumentId() != null ? request.getDocumentId() : UUID.randomUUID().toString();

        // 1. 文档分块（父子索引）
        List<DocumentChunk> chunks = documentChunkService.chunkDocument(
                documentId, request.getContent(), request.getMetadata());

        // 2. 批量获取Embedding（分批处理，每批最多100条）
        List<DocumentChunk> allChunks = new ArrayList<>(chunks);
        int batchSize = 100;
        for (int i = 0; i < allChunks.size(); i += batchSize) {
            int end = Math.min(i + batchSize, allChunks.size());
            List<DocumentChunk> batch = allChunks.subList(i, end);

            List<String> texts = batch.stream()
                    .map(DocumentChunk::getContent)
                    .toList();

            List<float[]> embeddings = llmClient.batchGetEmbedding(texts);

            for (int j = 0; j < batch.size(); j++) {
                batch.get(j).setEmbedding(embeddings.get(j));
            }
        }

        // 3. 批量写入Milvus
        milvusVectorService.batchInsert(allChunks);

        // 4. 批量写入Elasticsearch
        List<Map<String, Object>> esDocs = allChunks.stream()
                .map(elasticsearchBm25Service::chunkToEsDoc)
                .toList();
        elasticsearchBm25Service.batchInsert(esDocs);

        Map<String, Object> result = new HashMap<>();
        result.put("documentId", documentId);
        result.put("parentChunkCount", documentChunkService.getParentChunks(allChunks).size());
        result.put("childChunkCount", documentChunkService.getChildChunks(allChunks).size());

        return ResponseEntity.ok(result);
    }

    /**
     * 删除文档
     * DELETE /api/rag/documents/{documentId}
     */
    @DeleteMapping("/documents/{documentId}")
    public ResponseEntity<Map<String, String>> deleteDocument(@PathVariable String documentId) throws IOException {
        milvusVectorService.deleteByDocumentId(documentId);
        elasticsearchBm25Service.deleteByDocumentId(documentId);

        return ResponseEntity.ok(Map.of("message", "文档删除成功: " + documentId));
    }

    /**
     * 获取短期记忆
     * GET /api/rag/memory/short-term/{sessionId}
     */
    @GetMapping("/memory/short-term/{sessionId}")
    public ResponseEntity<List<String>> getShortTermMemory(@PathVariable String sessionId) {
        List<String> memories = memoryService.getShortTermMemoryTexts(sessionId);
        return ResponseEntity.ok(memories);
    }

    /**
     * 清除短期记忆
     * DELETE /api/rag/memory/short-term/{sessionId}
     */
    @DeleteMapping("/memory/short-term/{sessionId}")
    public ResponseEntity<Map<String, String>> clearShortTermMemory(@PathVariable String sessionId) {
        memoryService.clearShortTermMemory(sessionId);
        return ResponseEntity.ok(Map.of("message", "短期记忆已清除"));
    }

    /**
     * 文档入库请求
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class DocumentIngestRequest {
        private String documentId;
        private String content;
        private Map<String, Object> metadata;
    }
}