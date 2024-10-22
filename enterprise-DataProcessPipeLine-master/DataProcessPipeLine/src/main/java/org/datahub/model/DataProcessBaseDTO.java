package org.datahub.model;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class DataProcessBaseDTO {
    private String id;
    private String version;
    private String doi;
    private String title;
    private String attribute1;
    private String attribute2;
    private String path;
    private String pdfPath;
    private String createDate;
    private String modifyTime;
}
