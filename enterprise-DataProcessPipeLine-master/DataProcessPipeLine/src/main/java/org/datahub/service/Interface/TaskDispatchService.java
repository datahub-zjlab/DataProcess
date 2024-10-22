package org.datahub.service.Interface;

import com.fly.framework.web.domain.DataResponse;
import org.datahub.domain.vo.TaskBatchVO;
import org.datahub.exception.JobNotInProgressException;
import org.datahub.exception.NoDataAvailableException;
import org.datahub.model.JobInfoDTO;

import java.util.List;

/*
 * 1.从全部任务中选取batch的任务返回
 * */
public interface TaskDispatchService {
    DataResponse handleGetBatchTask(String jobId) throws JobNotInProgressException, NoDataAvailableException;

    List<TaskBatchVO> getDataFromClickhouse(String version, int offset, int limit);

    int getCountFromClickhouse(String version);

    JobInfoDTO getJobInfo(String jobId);
}
