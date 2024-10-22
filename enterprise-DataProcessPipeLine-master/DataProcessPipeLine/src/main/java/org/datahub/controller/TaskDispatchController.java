package org.datahub.controller;

import org.datahub.exception.ConcurrentProcessingException;
import org.datahub.exception.JobNotInProgressException;
import org.datahub.exception.NoDataAvailableException;
import org.datahub.service.Interface.TaskDispatchService;
import org.datahub.service.imp.TokenService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.fly.framework.web.domain.DataResponse;
import org.springframework.web.bind.annotation.RequestParam;

/*
 * 接收任务batch请求和处理参数信息请求
 * */
@RestController
@RequestMapping("/taskDispatch")
public class TaskDispatchController {
    @Autowired
    private TaskDispatchService taskDispatchService;

    @Autowired
    private TokenService tokenService;

    @RequestMapping("/getBatchTask")
    public DataResponse getBatchTask(@RequestParam String jobId) {
        try {
            return taskDispatchService.handleGetBatchTask(jobId);
        } catch (ConcurrentProcessingException | JobNotInProgressException | NoDataAvailableException e) {
            return DataResponse.error(e.getMessage());
        }
    }

    @RequestMapping("/getProcessInfo")
    public DataResponse getProcessInfo() {
        String accessToken = tokenService.getAccessToken();
        return DataResponse.success("successfully", accessToken);
    }
}
