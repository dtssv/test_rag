package com.rag.model;

import java.util.Map;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 文档分块模型，支持父子索引结构
 * - Parent Chunk: 大块，用于最终上下文供给
 * - Child Chunk: 小块，用于精确检索
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DocumentChunk {

    /**
     * 分块唯一ID
     */
    private String chunkId;

    /**
     * 父块ID（若当前为子块则有值，若为父块则为null）
     */
    private String parentChunkId;

    /**
     * 原始文档ID
     */
    private String documentId;

    /**
     * 分块类型: PARENT / CHILD
     */
    private ChunkType chunkType;

    /**
     * 分块内容文本
     */
    private String content;

    /**
     * 在原文中的起始位置
     */
    private int startPosition;

    /**
     * 在原文中的结束位置
     */
    private int endPosition;

    /**
     * 向量嵌入
     */
    private float[] embedding;

    /**
     * 元数据
     */
    private Map<String, Object> metadata;

    public enum ChunkType {
        PARENT, CHILD
    }
}