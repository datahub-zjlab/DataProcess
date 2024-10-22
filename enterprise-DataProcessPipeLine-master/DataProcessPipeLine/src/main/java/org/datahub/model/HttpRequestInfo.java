package org.datahub.model;

import lombok.Data;
import lombok.ToString;
import org.datahub.model.Enum.CleanMethod;
import org.datahub.model.Enum.MarkMethod;
import org.datahub.model.Enum.ParseMethod;

import java.util.Map;

/*
* http请求参数信息
* */
@Data
@ToString
public class HttpRequestInfo {
    String jobName;
    String jobParams;
    String outputPath;
    int userId;
    String dataSetID;
    int jobType;
}
