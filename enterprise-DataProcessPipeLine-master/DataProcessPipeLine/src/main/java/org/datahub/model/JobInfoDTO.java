package org.datahub.model;

import lombok.Data;

import java.sql.Timestamp;

/*
 * 数据库任务row
 *
 * */
@Data
public class JobInfoDTO {
    private long id;
    private String job_name;
    private String job_params;
    private String output_path;
    private String dataset_id;
    private String description;
    private int total_task;
    private int success_task;
    private int fail_task;
    private int total_pod_num;
    private int exit_pod_num;
    private long timeout;
    private int cluster_id;
    private int job_status;
    private int job_type;
    private Timestamp start_time;
    private Timestamp end_time;
    private String job_result;
    private int task_user;
    private Timestamp create_time;
    private Timestamp update_time;
    private byte deleted;
    private Timestamp delete_time;
}
