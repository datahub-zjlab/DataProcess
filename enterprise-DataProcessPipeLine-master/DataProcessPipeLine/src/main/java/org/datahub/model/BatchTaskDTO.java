package org.datahub.model;

import lombok.Data;

@Data
public class BatchTaskDTO {
    int jobID;
    long startTaskNo;
    long endTaskNo;
}
