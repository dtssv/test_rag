package com.rag.model;

import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * RAG请求模型
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RagRequest {

    /**
     * 用户会话ID
     */
    @NotBlank(message = "sessionId不能为空")
    private String sessionId;

    /**
     * 用户输入问题
     */
    @NotBlank(message = "query不能为空")
    private String query;

    /**
     * 是否强制使用查询扩写
     */
    @Builder.Default
    private boolean forceExpansion = false;

    /**
     * 是否启用rerank
     */
    @Builder.Default
    private boolean enableRerank = true;

    /**
     * 返回的最终文档数量
     */
    @Builder.Default
    private int topK = 5;
}