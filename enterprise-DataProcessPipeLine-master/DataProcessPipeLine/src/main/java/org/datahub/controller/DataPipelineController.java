package org.datahub.controller;

import org.datahub.service.imp.OSSDownloadService;
import org.springframework.core.io.Resource;
import com.fly.framework.web.domain.DataResponse;
import lombok.extern.slf4j.Slf4j;
import org.datahub.model.Enum.JobStatus;
import org.datahub.model.HttpRequestInfo;
import org.datahub.model.JobInfoDTO;
import org.datahub.model.ProcessConfig;
import org.datahub.service.imp.DataPipeLineService;
import org.datahub.service.imp.TaskScheduleService;
import org.datahub.utils.Imp.MySqlService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/*
 * 接收http数据处理任务和参数，进行处理
 * */
@Slf4j
@RestController
public class DataPipelineController {

    @Autowired
    private DataPipeLineService dataPipeLineService;

    @Autowired
    private MySqlService mySqlService;

    @Autowired
    private TaskScheduleService taskScheduleService;

    @Autowired
    private OSSDownloadService ossDownloadService;

    @RequestMapping("/test")
    public static String show(String[] args) {
        System.out.println("hello.spring-boot project！");
        return "hello.spring-boot project！";
    }

    @RequestMapping(value = "/DataPipeLine/CreateJob")
    public DataResponse CreateJob(@RequestBody HttpRequestInfo httpRequestInfo) {
        //1.请求信息插入数据库，获取jobID。
        //2.启动Pod
        JobInfoDTO jobInfo = new JobInfoDTO();
        jobInfo.setJob_name(httpRequestInfo.getJobName());
        jobInfo.setJob_params(httpRequestInfo.getJobParams());
        jobInfo.setOutput_path(httpRequestInfo.getOutputPath());
        jobInfo.setJob_status(JobStatus.RUNNING.getIndex());
        jobInfo.setTask_user(httpRequestInfo.getUserId());
        jobInfo.setJob_type(httpRequestInfo.getJobType());
        jobInfo.setDataset_id(httpRequestInfo.getDataSetID());
        if (jobInfo.getJob_name() == null || jobInfo.getJob_params() == null ||
                jobInfo.getTask_user() == 0 || jobInfo.getOutput_path() == null) {
            return DataResponse.error("启动失败,传入参数丢失或错误！");
        }
        ProcessConfig processConfig = new ProcessConfig();
        String exceptionInfo = null;
        try {
            Lock lock = new ReentrantLock();
            lock.lock();
            mySqlService.save(jobInfo);
            processConfig.setJobID(mySqlService.queryJobID());
            processConfig.setDataSetID(jobInfo.getDataset_id());
            processConfig.setJobType(jobInfo.getJob_type());
            processConfig.setOutputPath(jobInfo.getOutput_path());
            lock.unlock();
            int clusterID = dataPipeLineService.run(processConfig);
            if (clusterID > 0) {
                return DataResponse.success("启动成功！" + processConfig.toString(), processConfig.getJobID());
            }
        } catch (Exception e) {
            log.error(String.format("JobProgressService Exception! Exception: %s", e.toString()));
            exceptionInfo = e.toString();
        }
        return DataResponse.error("启动失败！" + exceptionInfo);
    }

    //    @RequestMapping(value = "/DataPipeLine/ReTryJob")
//    public String ReTryJob(@RequestParam("dataName") String dataName, @RequestParam("outputPath") String outputPath) {
//        return null;
//    }
    @RequestMapping(value = "/DataPipeLine/ReTryJob")
    public DataResponse ReTryJob(@RequestParam("jobID") Long jobID) {
        String exceptionInfo = null;
        ProcessConfig processConfig = new ProcessConfig();
        try {
            JobInfoDTO jobInfo = mySqlService.queryJobByID(jobID);
            processConfig.setJobID(jobID);
            processConfig.setDataSetID(jobInfo.getDataset_id());
            int clusterID = dataPipeLineService.run(processConfig);
            if (clusterID > 0) {
                return DataResponse.success("启动成功！" + processConfig, jobID);
            }
        } catch (Exception e) {
            log.error(String.format("JobProgressService Exception! Exception: %s", e.toString()));
            exceptionInfo = e.toString();
        }
        return DataResponse.error("启动失败！" + exceptionInfo);
    }

    @RequestMapping(value = "/DataPipeLine/StopJob")
    public DataResponse StopJob(@RequestParam("jobID") Long jobID) {
        //停止pod，更新数据库mysql
        String exceptionInfo = null;
        try {
            boolean isSuccess = dataPipeLineService.stop(jobID);
            if (isSuccess) {
                return DataResponse.success("停止成功！");
            }
        } catch (Exception e) {
            log.error(String.format("JobProgressService Exception! Exception: %s", e.toString()));
            exceptionInfo = e.toString();
        }
        return DataResponse.error("停止失败！" + exceptionInfo);
    }

    @GetMapping("/DataPipeLine/Download")
    @ResponseBody
    public ResponseEntity<Resource> DownloadPDF(@RequestParam("fileName") String fileName, @RequestParam("fileType") String fileType) {
        try {
            if (fileName == null) {
                fileName = "basic/ExampleForTest/";
            }
            StringBuilder name = new StringBuilder(fileName);
            Resource resource = ossDownloadService.downloadOssFile(name, fileType);
            if (resource.exists() || resource.isReadable()) {
                return ResponseEntity.ok()
                        .contentType(MediaType.APPLICATION_OCTET_STREAM)
                        .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + name + "\"")
                        .body(resource);
            } else {
                ResponseEntity.notFound().build();
            }
        } catch (Exception e) {
            log.error("Download File Error.Exception:" + e);
        }
        return ResponseEntity.status(500).build();
    }

    @RequestMapping(value = "/DataPipeLine/StartWatch")
    public DataResponse StartWatch() {
        String exceptionInfo = null;
        try {
            if (!taskScheduleService.isRunning()) {
                taskScheduleService.StartWatch();
                log.info("Start taskScheduleService successful.");
                return DataResponse.success("定时监控服务启动成功！");
            } else {
                log.info("TaskScheduleService is running.");
                return DataResponse.success("定时监控服务已在运行！");
            }
        } catch (Exception e) {
            log.error(String.format("Start taskScheduleService Exception.! Exception: %s", e));
            exceptionInfo = e.toString();
        }
        return DataResponse.error("定时监控服务启动失败！" + exceptionInfo);
    }
}
