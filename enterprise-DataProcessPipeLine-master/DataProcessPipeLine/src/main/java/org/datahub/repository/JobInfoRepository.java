package org.datahub.repository;

import org.datahub.model.JobInfo;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

@Repository
public interface JobInfoRepository extends JpaRepository<JobInfo, Long> {
    @Modifying
    @Transactional
    @Query("UPDATE JobInfo j SET j.jobResult = :jobResult WHERE j.id = :jobId")
    void updateJobResult(@Param("jobId") long jobId, @Param("jobResult") String jobResult);
}
