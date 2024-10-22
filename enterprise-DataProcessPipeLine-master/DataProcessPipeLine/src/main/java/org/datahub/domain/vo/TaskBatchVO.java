package org.datahub.domain.vo;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
@AllArgsConstructor
public class TaskBatchVO {
    private String id;
    private String version;
    private String doi;
    private String path;
    private String pdfPath;
    private String attribute2;
    private String fileExtension;  // 新增解析出的 extension 属性
    private String language;  // 新增解析出的 extension 属性
}
