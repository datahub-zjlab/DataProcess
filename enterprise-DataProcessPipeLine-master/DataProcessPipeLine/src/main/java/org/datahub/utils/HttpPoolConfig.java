package org.datahub.utils;

public class HttpPoolConfig {
    /** http连接池大小 */
    public int httpPoolSize;
    /** http连接超时时间 */
    public int httpConnectTimeout;
    /** http连接池等待超时时间 */
    public int httpWaitTimeout;
    /** http响应包间隔超时时间 */
    public int httpSocketTimeout;
    /** http重试次数 */
    public int httpRetryCount;
    /** http重试间隔时间 */
    public int httpRetryInterval;
    /** http监控间隔时间 定时清理 打印连接池状态 */
    public int httpMonitorInterval;
    /** http关闭空闲连接的等待时间 */
    public int httpCloseIdleConnectionWaitTime;
}
