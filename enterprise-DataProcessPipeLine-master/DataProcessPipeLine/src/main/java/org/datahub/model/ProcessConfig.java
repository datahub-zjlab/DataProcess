package org.datahub.model;

import lombok.Data;
import org.datahub.model.Enum.CleanMethod;
import org.datahub.model.Enum.MarkMethod;
import org.datahub.model.Enum.ParseMethod;

import java.util.Map;
/*
* 内部传递参数
* 启动pod时注入pod的参数
* */
@Data
public class ProcessConfig {
    long jobID;
    String dataSetID;
    Map<String,String> fileterRule;
    String outputPath;
    CleanMethod cleanMethod;
    MarkMethod markMethod;
    ParseMethod parseMethod;
    int jobType;
}
