package org.datahub.utils;

import lombok.extern.slf4j.Slf4j;
import org.apache.http.HttpEntityEnclosingRequest;
import org.apache.http.HttpRequest;
import org.apache.http.NoHttpResponseException;
import org.apache.http.client.HttpRequestRetryHandler;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.protocol.HttpClientContext;
import org.apache.http.config.Registry;
import org.apache.http.config.RegistryBuilder;
import org.apache.http.conn.socket.ConnectionSocketFactory;
import org.apache.http.conn.socket.LayeredConnectionSocketFactory;
import org.apache.http.conn.ssl.SSLConnectionSocketFactory;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.impl.conn.PoolingHttpClientConnectionManager;
import org.apache.http.pool.PoolStats;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.stereotype.Component;

import javax.net.ssl.SSLException;
import java.io.InterruptedIOException;
import java.net.UnknownHostException;
import java.util.TimerTask;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

@Slf4j
@Component
public class HttpsPool implements InitializingBean {

    private CloseableHttpClient httpClient;
    private ScheduledExecutorService monitorExecutor;

    /**
     * 连接池管理器
     */
    private PoolingHttpClientConnectionManager buildHttpManger(HttpPoolConfig httpPoolConfig) {
        LayeredConnectionSocketFactory sslSocketFactory = SSLConnectionSocketFactory.getSocketFactory();
        Registry<ConnectionSocketFactory> registry = RegistryBuilder.<ConnectionSocketFactory>create()
                .register("https", sslSocketFactory).build();
        PoolingHttpClientConnectionManager manager = new PoolingHttpClientConnectionManager(registry);
        manager.setMaxTotal(httpPoolConfig.httpPoolSize);
        manager.setDefaultMaxPerRoute(httpPoolConfig.httpPoolSize);
        return manager;
    }

    /**
     * 建立httpClient
     */
    private CloseableHttpClient buildHttpClient(HttpPoolConfig httpPoolConfig, PoolingHttpClientConnectionManager manager) {
        // 请求配置
        RequestConfig config = RequestConfig.custom()
                .setConnectTimeout(httpPoolConfig.httpConnectTimeout)
                .setSocketTimeout(httpPoolConfig.httpSocketTimeout)
                .setConnectionRequestTimeout(httpPoolConfig.httpWaitTimeout)
                .build();
        // 失败重试机制
        HttpRequestRetryHandler retryHandler = (e, c, context) -> {
            if (c > httpPoolConfig.httpRetryCount) {
                log.error("HttpPool request retry more than {} times", httpPoolConfig.httpRetryCount, e);
                return false;
            }
            if (e == null) {
                log.info("HttpPool request exception is null.");
                return false;
            }
            if (e instanceof NoHttpResponseException) {
                //服务器没有响应,可能是服务器断开了连接,应该重试
                log.error("HttpPool receive no response from server, retry");
                return true;
            }
            // SSL握手异常
            if (e instanceof InterruptedIOException // 超时
                    || e instanceof UnknownHostException // 未知主机
                    || e instanceof SSLException) { // SSL异常
                log.error("HttpPool request error, retry", e);
                return true;
            } else {
                log.error("HttpPool request unknown error, retry", e);
            }
            // 对于关闭连接的异常不进行重试
            HttpClientContext clientContext = HttpClientContext.adapt(context);
            HttpRequest request = clientContext.getRequest();
            return !(request instanceof HttpEntityEnclosingRequest);
        };
        // 构建httpClient
        return HttpClients.custom().setDefaultRequestConfig(config)
                .setConnectionManager(manager).setRetryHandler(retryHandler).build();
    }

    /**
     * 建立连接池监视器
     */
    private ScheduledExecutorService buildMonitorExecutor(HttpPoolConfig httpPoolConfig,
                                                          PoolingHttpClientConnectionManager manager) {
        TimerTask timerTask = new TimerTask() {
            @Override
            public void run() {
                // 关闭过期连接
                manager.closeExpiredConnections();
                // 关闭空闲时间超过一定时间的连接
                manager.closeIdleConnections(httpPoolConfig.httpCloseIdleConnectionWaitTime, TimeUnit.MILLISECONDS);
                // 打印连接池状态
                PoolStats poolStats = manager.getTotalStats();
                // max:最大连接数, available:可用连接数, leased:已借出连接数, pending:挂起(表示当前等待从连接池中获取连接的线程数量)
//                log.info("HttpPool status {}", poolStats);
            }
        };
        ScheduledExecutorService executor = Executors.newScheduledThreadPool(1);
        int time = httpPoolConfig.httpMonitorInterval;
        executor.scheduleAtFixedRate(timerTask, time, time, TimeUnit.MILLISECONDS);
        return executor;
    }

    /**
     * 关闭连接池
     */
    public void close() {
        try {
            httpClient.close();
            monitorExecutor.shutdown();
        } catch (Exception e) {
            log.error("HttpPool close http client error", e);
        }
    }

    /**
     * 发起get请求
     */
    public String get(String url) { return DoHttp.get(httpClient, url); }

    /**
     * 发起json格式的post请求
     */
    public void sendPost(String url, String json) { DoHttp.jsonPost(httpClient, url, json); }

    /**
     * 初始化连接池
     */
    @Override
    public void afterPropertiesSet() throws Exception {
        HttpPoolConfig config = new HttpPoolConfig();
        config.httpPoolSize = 20;
        config.httpConnectTimeout = 1000; // 一秒
        config.httpRetryCount = 1;
        config.httpRetryInterval = 500;
        config.httpCloseIdleConnectionWaitTime = 60 * 1000; // 一分钟
        config.httpMonitorInterval = 30000;
        config.httpSocketTimeout = 2 * 1000;
        config.httpWaitTimeout = 2 * 1000;
        PoolingHttpClientConnectionManager manager = buildHttpManger(config);
        httpClient = buildHttpClient(config, manager);
        monitorExecutor = buildMonitorExecutor(config, manager);
    }

}

