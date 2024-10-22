package org.datahub.model.Enum;

import lombok.Getter;

@Getter
public enum JobStatus {
    RUNNING(10,"Running"),
    STOP(20,"Stop");

    JobStatus(int index, String msg){
        this.index = index;
        this.msg = msg;
    }

    private final int index;
    private final String msg;

}
