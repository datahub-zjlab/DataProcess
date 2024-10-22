package org.datahub.model;

import lombok.Data;

import javax.persistence.*;
import java.sql.Timestamp;

@Data
@Entity
@Table(name = "job_info")
public class JobInfo {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private long id;

    @Column(name = "job_name")
    private String jobName;

    @Column(name = "job_params")
    private String jobParams;

    @Column(name = "output_path")
    private String outputPath;

    @Column(name = "dataset_id")
    private String datasetId;

    @Column(name = "description")
    private String description;

    @Column(name = "total_task")
    private int totalTask;

    @Column(name = "success_task")
    private int successTask;

    @Column(name = "fail_task")
    private int failTask;

    @Column(name = "total_pod_num")
    private int totalPodNum;

    @Column(name = "exit_pod_num")
    private int exitPodNum;

    @Column(name = "timeout")
    private long timeout;

    @Column(name = "cluster_id")
    private int clusterId;

    @Column(name = "job_status")
    private int jobStatus;

    @Column(name = "job_type")
    private int jobType;

    @Column(name = "start_time")
    private Timestamp startTime;

    @Column(name = "end_time")
    private Timestamp endTime;

    @Column(name = "job_result")
    private String jobResult;

    @Column(name = "task_user")
    private int taskUser;

    @Column(name = "create_time")
    private Timestamp createTime;

    @Column(name = "update_time")
    private Timestamp updateTime;

    @Column(name = "deleted")
    private byte deleted;

    @Column(name = "delete_time")
    private Timestamp deleteTime;
}
