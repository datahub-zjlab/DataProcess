package org.datahub.utils;

import lombok.extern.slf4j.Slf4j;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.client.methods.HttpRequestBase;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.util.EntityUtils;

@Slf4j
public class DoHttp {
    public static String get(CloseableHttpClient httpClient,String url){
        HttpGet httpGet=new HttpGet(url);
        return doRequest(httpClient,url,httpGet);

    }

    public static String jsonPost(CloseableHttpClient httpClient,String url,String json){
        HttpPost httpPost = new HttpPost(url);
        httpPost.setHeader("Content-Type", "application/json;charset=UTF-8");
        StringEntity entity = new StringEntity(json,"UTF-8");
        httpPost.setEntity(entity);
        return doRequest(httpClient,url,httpPost);
    }

    public static String doRequest(CloseableHttpClient httpClient, String url, HttpRequestBase httpRequest){
        try(CloseableHttpResponse httpResponse=httpClient.execute(httpRequest)){
            int code = httpResponse.getStatusLine().getStatusCode();
            HttpEntity responseEntity = httpResponse.getEntity();
            String responseBody = null;
            if(responseEntity!=null){
                responseBody= EntityUtils.toString(responseEntity);
            }
            if(code !=200){
                log.error("http post error, url: {}, code: {}, result: {}", url, code, responseBody);
                return null;
            }
            return responseBody;
        }
        catch (Exception e){
            log.error("http post error, url: {}", url, e);
        }
        return null;
    }
}
