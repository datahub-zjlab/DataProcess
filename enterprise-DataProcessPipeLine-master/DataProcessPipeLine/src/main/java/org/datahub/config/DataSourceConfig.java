package org.datahub.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.jdbc.DataSourceBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.jdbc.core.JdbcTemplate;
import ru.yandex.clickhouse.ClickHouseDataSource;
import ru.yandex.clickhouse.settings.ClickHouseProperties;
import org.springframework.beans.factory.annotation.Value;

import javax.sql.DataSource;

@Configuration
@Slf4j
public class DataSourceConfig {
    @Value("${spring.datasource.clickhouse.jdbc-url}")
    private String jdbcUrl;
    @Value("${spring.datasource.clickhouse.username}")
    private String username;
    @Value("${spring.datasource.clickhouse.password}")
    private String password;
    @Value("${spring.datasource.clickhouse.properties.socket_timeout}")
    private Integer socketTimeout;
    @Value("${spring.datasource.clickhouse.properties.max_execution_time}")
    private Integer maxExecutionTime;

    @Primary
    @Bean(name = "mysqlDataSource")
    @Qualifier("mysqlDataSource")
    @ConfigurationProperties(prefix = "spring.datasource.mysql")
    public DataSource mysqlDataSource(){
        return DataSourceBuilder.create().build();
    }

    @Bean(name = "clickhouseDataSource")
    @Qualifier("clickhouseDataSource")
    @ConfigurationProperties(prefix = "spring.datasource.clickhouse")
    public DataSource clickhouseDataSource(){
        ClickHouseProperties properties = new ClickHouseProperties();
        properties.setSocketTimeout(socketTimeout); // 设置socket超时
        properties.setConnectionTimeout(60000); // 设置连接超时
        properties.setDataTransferTimeout(60000); // 设置数据传输超时
        properties.setMaxExecutionTime(maxExecutionTime); //设置最大执行时间
        properties.setUser(username); // 设置用户名
        properties.setPassword(password); // 设置密码

        ClickHouseDataSource dataSource = new ClickHouseDataSource(jdbcUrl, properties);

        return dataSource;
    }

    @Bean(name="mysqlJdbcTemplate")
    public JdbcTemplate mysqlJdbcTemplate (
            @Qualifier("mysqlDataSource") DataSource dataSource ) {
        return new JdbcTemplate(dataSource);
    }
    @Bean(name="clickhouseJdbcTemplate")
    public JdbcTemplate clickhouseJdbcTemplate(
            @Qualifier("clickhouseDataSource") DataSource dataSource) {
        JdbcTemplate jdbcTemplate = new JdbcTemplate(dataSource);
        return jdbcTemplate;
    }
}
