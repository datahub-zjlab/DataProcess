package org.datahub.domain.enums;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum JobTypeEnum {

    /**
     * 其他
     */
    Normal(0, "normal"),

    /**
     * arxiv
     */
    ARXIV(1, "pdfParse"),
    /**
     * cc
     */
    CC(2, "cc"),
    /**
     * parquet
     */
    PARQUET(3, "crawler");

    private final int code;

    private final String name;

    public static JobTypeEnum parse(int code) {
        for (JobTypeEnum typeEnum : JobTypeEnum.values()) {
            if (typeEnum.getCode() == code) {
                return typeEnum;
            }
        }
        return null;
    }
}