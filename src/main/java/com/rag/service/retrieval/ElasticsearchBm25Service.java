package com.rag.service.retrieval;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import javax.annotation.PostConstruct;

import co.elastic.clients.elasticsearch._types.FieldValue;
import org.springframework.stereotype.Service;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import co.elastic.clients.elasticsearch.core.search.Hit;
import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch._types.query_dsl.BoolQuery;
import co.elastic.clients.elasticsearch._types.query_dsl.MatchQuery;
import co.elastic.clients.elasticsearch.indices.CreateIndexRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import com.rag.config.ElasticsearchConfig;

/**
 * Elasticsearch BM25检索服务
 * 负责文档块的全文检索
 * 支持父子索引：子块用于BM25检索
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ElasticsearchBm25Service {

    private final ElasticsearchClient esClient;
    private final ElasticsearchConfig esConfig;

    private static final String FIELD_CHUNK_ID = "chunk_id";
    private static final String FIELD_PARENT_CHUNK_ID = "parent_chunk_id";
    private static final String FIELD_DOCUMENT_ID = "document_id";
    private static final String FIELD_CHUNK_TYPE = "chunk_type";
    private static final String FIELD_CONTENT = "content";

    /**
     * 应用启动时初始化索引
     */
    @PostConstruct
    public void init() {
        try {
            ensureIndexExists();
            log.info("Elasticsearch索引[{}]初始化完成", esConfig.getIndexName());
        } catch (Exception e) {
            log.warn("Elasticsearch初始化失败(服务可能未启动): {}", e.getMessage());
        }
    }

    /**
     * 确保索引存在
     */
    private void ensureIndexExists() throws IOException {
        String indexName = esConfig.getIndexName();

        try {
            esClient.indices().create(new CreateIndexRequest.Builder()
                    .index(indexName)
                    .mappings(m -> m
                            .properties(FIELD_CHUNK_ID, p -> p.keyword(k -> k))
                            .properties(FIELD_PARENT_CHUNK_ID, p -> p.keyword(k -> k))
                            .properties(FIELD_DOCUMENT_ID, p -> p.keyword(k -> k))
                            .properties(FIELD_CHUNK_TYPE, p -> p.keyword(k -> k))
                            .properties(FIELD_CONTENT, p -> p.text(t -> t
                                    .analyzer("ik_max_word")
                                    .searchAnalyzer("ik_smart"))))
                    .build());
            log.info("创建Elasticsearch索引[{}]成功", indexName);
        } catch (Exception e) {
            log.debug("索引[{}]可能已存在: {}", indexName, e.getMessage());
        }
    }

    /**
     * 批量插入文档块到ES
     *
     * @param chunks 文档块列表
     */
    public void batchInsert(List<Map<String, Object>> chunks) throws IOException {
        if (chunks == null || chunks.isEmpty())
            return;

        String indexName = esConfig.getIndexName();

        for (Map<String, Object> chunk : chunks) {
            String chunkId = (String) chunk.get(FIELD_CHUNK_ID);
            esClient.index(i -> i
                    .index(indexName)
                    .id(chunkId)
                    .document(chunk));
        }

        // 刷新索引使文档可搜索
        esClient.indices().refresh(r -> r.index(indexName));
        log.info("批量插入{}个文档块到Elasticsearch", chunks.size());
    }

    /**
     * 将DocumentChunk转换为ES文档Map
     */
    public Map<String, Object> chunkToEsDoc(com.rag.model.DocumentChunk chunk) {
        Map<String, Object> doc = new HashMap<>();
        doc.put(FIELD_CHUNK_ID, chunk.getChunkId());
        doc.put(FIELD_PARENT_CHUNK_ID, chunk.getParentChunkId() != null ? chunk.getParentChunkId() : "");
        doc.put(FIELD_DOCUMENT_ID, chunk.getDocumentId());
        doc.put(FIELD_CHUNK_TYPE, chunk.getChunkType().name());
        doc.put(FIELD_CONTENT, chunk.getContent());
        return doc;
    }

    /**
     * BM25检索
     *
     * @param query     查询文本
     * @param topK      返回数量
     * @param chunkType 块类型过滤
     * @return 检索结果列表
     */
    public List<Bm25SearchResult> search(String query, int topK, String chunkType) throws IOException {
        String indexName = esConfig.getIndexName();

        // 构建布尔查询：内容匹配 + 块类型过滤
        BoolQuery boolQuery = BoolQuery.of(b -> {
            b.must(MatchQuery.of(m -> m
                    .field(FIELD_CONTENT)
                    .query(query))._toQuery());

            if (chunkType != null && !chunkType.isEmpty()) {
                b.filter(f -> f
                        .term(t -> t
                                .field(FIELD_CHUNK_TYPE)
                                .value(chunkType)));
            }
            return b;
        });

        SearchResponse<Map> searchResponse = esClient.search(s -> s
                .index(indexName)
                .query(boolQuery._toQuery())
                .size(topK),
                Map.class);

        List<Bm25SearchResult> results = new ArrayList<>();
        for (Hit<Map> hit : searchResponse.hits().hits()) {
            Map<String, Object> source = hit.source();
            if (source != null) {
                Bm25SearchResult result = Bm25SearchResult.builder()
                        .chunkId((String) source.get(FIELD_CHUNK_ID))
                        .parentChunkId((String) source.get(FIELD_PARENT_CHUNK_ID))
                        .documentId((String) source.get(FIELD_DOCUMENT_ID))
                        .content((String) source.get(FIELD_CONTENT))
                        .score(hit.score() != null ? hit.score() : 0.0)
                        .build();
                results.add(result);
            }
        }

        log.debug("BM25检索返回{}个结果", results.size());
        return results;
    }

    /**
     * 根据chunkId批量查询父块内容
     */
    public Map<String, String> batchGetParentContent(List<String> parentChunkIds) throws IOException {
        if (parentChunkIds == null || parentChunkIds.isEmpty()) {
            return Collections.emptyMap();
        }

        String indexName = esConfig.getIndexName();

        // 使用terms查询批量获取
        SearchResponse<Map> searchResponse = esClient.search(s -> s
                .index(indexName)
                .size(parentChunkIds.size())
                .query(q -> q
                        .bool(b -> b
                                .must(m -> m
                                        .terms(t -> t
                                                .field(FIELD_CHUNK_ID)
                                                .terms(terms -> terms
                                                        .value(parentChunkIds.stream().map(FieldValue::of)
                                                                .collect(Collectors.toList())))))
                                .filter(f -> f
                                        .term(t -> t
                                                .field(FIELD_CHUNK_TYPE)
                                                .value("PARENT"))))),
                Map.class);

        Map<String, String> parentContentMap = new HashMap<>();
        for (Hit<Map> hit : searchResponse.hits().hits()) {
            Map<String, Object> source = hit.source();
            if (source != null) {
                String chunkId = (String) source.get(FIELD_CHUNK_ID);
                String content = (String) source.get(FIELD_CONTENT);
                parentContentMap.put(chunkId, content);
            }
        }

        return parentContentMap;
    }

    /**
     * 删除指定文档的所有块
     */
    public void deleteByDocumentId(String documentId) throws IOException {
        String indexName = esConfig.getIndexName();

        esClient.deleteByQuery(d -> d
                .index(indexName)
                .query(q -> q
                        .term(t -> t
                                .field(FIELD_DOCUMENT_ID)
                                .value(documentId))));
        log.info("删除文档[{}]的所有BM25索引数据", documentId);
    }

    /**
     * BM25检索结果
     */
    @lombok.Data
    @lombok.Builder
    @lombok.NoArgsConstructor
    @lombok.AllArgsConstructor
    public static class Bm25SearchResult {
        private String chunkId;
        private String parentChunkId;
        private String documentId;
        private String content;
        private double score;
    }
}