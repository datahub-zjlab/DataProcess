package org.datahub.controller;

import com.fly.framework.web.domain.DataResponse;
import lombok.extern.slf4j.Slf4j;
import org.datahub.model.TaskResultDTO;
import org.datahub.service.imp.JobProgressService;
import org.datahub.service.imp.RecResultService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/*
 * 接收pod处理结果，并存入数据库
 * */
@RestController
@Slf4j
public class RecResultController {
    @Autowired
    private RecResultService recResultService;

    @Autowired
    private JobProgressService jobProgressService;

    @RequestMapping(value = "/DataPipeLine/PutPodResult")
    public DataResponse RecPodResult(@RequestBody TaskResultDTO taskResultDTO) {
        recResultService.run(taskResultDTO);
        return DataResponse.success("收到信息！" );
    }

    @RequestMapping(value = "/DataPipeLine/GetJobProgress",method = RequestMethod.GET)
    public DataResponse GetJobProgress(@RequestParam("userID") int userID) {
        Object result = jobProgressService.run(userID);
        return DataResponse.success("查询用户进度结果",result);
    }
}
