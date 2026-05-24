package com.rag.service.chunk;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import com.rag.config.RagConfig;
import com.rag.model.DocumentChunk;
import com.rag.model.DocumentChunk.ChunkType;

/**
 * 文档分块服务
 * 实现父子索引/大小块嵌套分块策略：
 * 1. 先将文档切成大块(Parent Chunk)，用于最终上下文供给
 * 2. 再将每个大块切成小块(Child Chunk)，用于精确检索
 * 3. 检索时先匹配小块，再通过parentChunkId找到对应大块返回
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class DocumentChunkService {

    private final RagConfig ragConfig;

    /**
     * 对文档进行父子索引分块
     *
     * @param documentId 文档ID
     * @param content    文档内容
     * @param metadata   元数据
     * @return 父子分块列表
     */
    public List<DocumentChunk> chunkDocument(String documentId, String content, Map<String, Object> metadata) {
        RagConfig.ChunkConfig chunkConfig = ragConfig.getChunk();
        List<DocumentChunk> allChunks = new ArrayList<>();

        // 第一步：切大块(Parent Chunk)
        List<DocumentChunk> parentChunks = splitIntoChunks(
                documentId, content, chunkConfig.getParentSize(),
                chunkConfig.getParentOverlap(), ChunkType.PARENT, metadata);

        // 第二步：对每个大块切小块(Child Chunk)
        for (DocumentChunk parentChunk : parentChunks) {
            List<DocumentChunk> childChunks = splitIntoChunks(
                    documentId, parentChunk.getContent(),
                    chunkConfig.getChildSize(), chunkConfig.getChildOverlap(),
                    ChunkType.CHILD, metadata);

            // 设置子块的父块ID
            for (DocumentChunk childChunk : childChunks) {
                childChunk.setParentChunkId(parentChunk.getChunkId());
                // 调整子块在原文中的位置
                childChunk.setStartPosition(parentChunk.getStartPosition() + childChunk.getStartPosition());
                childChunk.setEndPosition(parentChunk.getStartPosition() + childChunk.getEndPosition());
            }

            allChunks.add(parentChunk);
            allChunks.addAll(childChunks);
        }

        log.info("文档[{}]分块完成: {}个父块, {}个子块",
                documentId,
                parentChunks.size(),
                allChunks.size() - parentChunks.size());

        return allChunks;
    }

    /**
     * 按固定大小+重叠切分文本
     */
    private List<DocumentChunk> splitIntoChunks(String documentId, String content,
            int chunkSize, int overlap,
            ChunkType chunkType,
            Map<String, Object> metadata) {
        List<DocumentChunk> chunks = new ArrayList<>();
        if (content == null || content.isEmpty()) {
            return chunks;
        }

        int contentLength = content.length();
        int start = 0;
        int chunkIndex = 0;

        do {
            int end = Math.min(start + chunkSize, contentLength);

            // 尝试在句子边界处切割
            if (end < contentLength) {
                int sentenceBreak = findSentenceBreak(content, start, end);
                if (sentenceBreak > start) {
                    end = sentenceBreak;
                }
            }

            String chunkContent = content.substring(start, end).trim();
            if (!chunkContent.isEmpty()) {
                DocumentChunk chunk = DocumentChunk.builder()
                        .chunkId(generateChunkId(documentId, chunkType, chunkIndex))
                        .parentChunkId(chunkType == ChunkType.PARENT ? null : "")
                        .documentId(documentId)
                        .chunkType(chunkType)
                        .content(chunkContent)
                        .startPosition(start)
                        .endPosition(end)
                        .metadata(metadata != null ? new HashMap<>(metadata) : new HashMap<>())
                        .build();
                chunks.add(chunk);
                chunkIndex++;
            }

            start = end - overlap;
        } while (start < contentLength);

        return chunks;
    }

    /**
     * 在指定范围内查找句子边界（优先在句号、换行等处切割）
     */
    private int findSentenceBreak(String content, int start, int end) {
        // 从end往回找句子边界
        for (int i = end - 1; i > start; i--) {
            char c = content.charAt(i);
            if (c == '。' || c == '！' || c == '？' || c == '\n' ||
                    c == '.' || c == '!' || c == '?') {
                return i + 1;
            }
        }
        // 没找到句子边界，找空格
        for (int i = end - 1; i > start; i--) {
            if (content.charAt(i) == ' ' || content.charAt(i) == '\t') {
                return i + 1;
            }
        }
        return end;
    }

    /**
     * 生成块ID
     */
    private String generateChunkId(String documentId, ChunkType chunkType, int index) {
        return String.format("%s_%s_%d", documentId, chunkType.name().toLowerCase(), index);
    }

    /**
     * 从分块列表中筛选出父块
     */
    public List<DocumentChunk> getParentChunks(List<DocumentChunk> chunks) {
        return chunks.stream()
                .filter(c -> c.getChunkType() == ChunkType.PARENT)
                .collect(Collectors.toList());
    }

    /**
     * 从分块列表中筛选出子块
     */
    public List<DocumentChunk> getChildChunks(List<DocumentChunk> chunks) {
        return chunks.stream()
                .filter(c -> c.getChunkType() == ChunkType.CHILD)
                .collect(Collectors.toList());
    }

    /**
     * 根据子块ID找到对应的父块
     */
    public Optional<DocumentChunk> findParentChunk(List<DocumentChunk> allChunks, String parentChunkId) {
        return allChunks.stream()
                .filter(c -> c.getChunkId().equals(parentChunkId))
                .findFirst();
    }
}