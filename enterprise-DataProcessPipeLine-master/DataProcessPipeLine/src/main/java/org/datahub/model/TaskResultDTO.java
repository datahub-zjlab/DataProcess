package org.datahub.model;

import lombok.Data;

@Data
public class TaskResultDTO {
    int jobID;

    int successTask;

    int failTask;

}
